"""Offline tooling safety regressions; Docker and network calls are mocked."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import materialize
import preflight
import runner


class Safeguards(unittest.TestCase):
    def sample(self, ram=8, disk=50):
        return {'available_ram_bytes': ram * preflight.GIB,
                'docker_memory_bytes': 16 * preflight.GIB,
                'docker_free_bytes': disk * preflight.GIB,
                'system_free_bytes': 15 * preflight.GIB}

    def test_role_specific_resource_boundaries(self):
        for role, ram, start_disk, continuous_disk in [('node', 5, 20, 15), ('python', 8, 50, 50)]:
            with self.subTest(role=role):
                self.assertFalse(preflight.problems(role, self.sample(ram, start_disk), start=True))
                self.assertIn('RAM', preflight.problems(role, self.sample(ram - 1, start_disk), start=True))
                self.assertIn('DOCKER_DISK', preflight.problems(role, self.sample(ram, start_disk - 1), start=True))
                self.assertFalse(preflight.problems(role, self.sample(5, continuous_disk), start=False))
                self.assertIn('DOCKER_DISK', preflight.problems(role, self.sample(5, continuous_disk - 1), start=False))
                self.assertIn('RAM', preflight.problems(role, self.sample(4, continuous_disk), start=False))

    def test_system_and_docker_memory_floors(self):
        data = self.sample()
        data['system_free_bytes'] -= 1
        data['docker_memory_bytes'] = 4 * preflight.GIB
        self.assertIn('SYSTEM_DISK', preflight.problems('node', data, start=True))
        self.assertIn('RAM', preflight.problems('node', data, start=True))

    def test_frozen_identities(self):
        self.assertEqual(materialize.COMMITS['node'], 'dfa3138f76bb25fa83d0ca61adf4cc256b53fa6c')
        self.assertEqual(materialize.COMMITS['python'], 'a76c6f47bde88f39570c99fb1de012cfbda9df72')
        self.assertEqual(materialize.BASE, 'ghcr.io/apol-bit/ispeak-python@sha256:a938fb342d2ef69ad0e6110b4031d41b7ddc9628311f2f50788922a2c034c05e')

    def test_disabled_execution_does_no_work(self):
        v = runner.Validation.__new__(runner.Validation)
        v.role = 'node'
        with patch.dict(os.environ, {'AUTOMATION_REVIEW_APPROVED': 'false'}, clear=False), patch.object(runner, 'enforce') as enforce:
            with self.assertRaises(RuntimeError):
                v.execute()
            enforce.assert_not_called()

    def test_testing_policy_is_required(self):
        v = runner.Validation.__new__(runner.Validation)
        v.role = 'node'
        with patch.dict(os.environ, {'AUTOMATION_REVIEW_APPROVED': 'true', 'TESTING_CONTAINMENT_ACCEPTED': 'false'}), patch.object(runner, 'enforce') as enforce:
            with self.assertRaises(RuntimeError):
                v.execute()
            enforce.assert_not_called()

    def test_private_access_stops_before_pull(self):
        v = runner.Validation.__new__(runner.Validation)
        v.command = Mock()
        v.token = 'synthetic-not-a-credential'
        with patch.dict(os.environ, {'PRIVATE_BASE_ACCESS_APPROVED': 'false'}):
            with self.assertRaises(RuntimeError):
                v.login()
            v.command.assert_not_called()

    def test_no_cancellation_fixture_method(self):
        self.assertFalse(hasattr(runner.Validation, 'cancel_probe'))

    def test_owned_absence_requires_successful_queries(self):
        v = runner.Validation.__new__(runner.Validation)
        v.builder = 'owned'
        v.evidence = Mock()
        v.docker = Mock(side_effect=[b'', b''])
        v.assert_builder_absent('after_cleanup')
        v.docker = Mock(side_effect=[b'', b'leftover'])
        with self.assertRaises(RuntimeError):
            v.assert_builder_absent('after_cleanup')
        v.docker = Mock(side_effect=RuntimeError('API failure'))
        with self.assertRaises(RuntimeError):
            v.assert_builder_absent('after_cleanup')

    def test_unowned_builder_is_never_stopped(self):
        v = runner.Validation.__new__(runner.Validation)
        v.builder, v.key = 'owned', '123-1'
        data = {'Name': '/some-other-container', 'Config': {'Env': []}}
        v.docker = Mock(side_effect=[b'id', json.dumps([data]).encode()])
        with self.assertRaises(RuntimeError):
            v.stop_builder()
        self.assertEqual(v.docker.call_count, 2)

    def test_node_gate_rejects_unreviewed_run_identity(self):
        from node_gate import require_node_success
        with patch.dict(os.environ, {'NODE_VALIDATION_RUN_ID': 'REVIEW_REQUIRED', 'NODE_AUTOMATION_SHA': 'REVIEW_REQUIRED'}):
            with self.assertRaises(RuntimeError):
                require_node_success()

    def test_builder_caps_and_live_parallelism(self):
        image = 'moby/buildkit@sha256:' + 'a' * 64
        valid = {'Id': 'owned-id', 'Image': 'image-id', 'RestartCount': 0,
                 'State': {'Running': True, 'OOMKilled': False},
                 'HostConfig': {'Memory': 4 * preflight.GIB, 'MemorySwap': 4 * preflight.GIB,
                                'CpuQuota': 200000, 'CpuPeriod': 100000, 'RestartPolicy': {'Name': 'no'}},
                 'Config': {'Env': ['ISPEAK_RELEASE_RUN=123-1'], 'Image': image}}
        for mutation in (None, 'Memory', 'MemorySwap', 'CpuQuota', 'restart', 'parallelism'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                metadata = copy.deepcopy(valid)
                if mutation in ('Memory', 'MemorySwap', 'CpuQuota'):
                    metadata['HostConfig'][mutation] = 0
                if mutation == 'restart':
                    metadata['HostConfig']['RestartPolicy']['Name'] = 'unless-stopped'
                parallelism = 2 if mutation == 'parallelism' else 1
                config = ('[worker.oci]\nenabled=true\nmax-parallelism=' + str(parallelism)
                          + '\n[worker.containerd]\nenabled=false\n').encode()
                v = runner.Validation.__new__(runner.Validation)
                v.root, v.builder, v.key = Path(temp), 'owned', '123-1'
                v.assert_builder_absent, v.evidence = Mock(), Mock()
                def docker(*args, **_kwargs):
                    if args[:2] == ('buildx', 'version'):
                        return b'github.com/docker/buildx v0.28.0 test'
                    if args[0] == 'version':
                        return b'28.0.4'
                    if args[0] == 'inspect':
                        return json.dumps([metadata]).encode()
                    if args[:2] == ('image', 'inspect'):
                        return json.dumps([{'Id': 'image-id'}]).encode()
                    if args[0] == 'exec':
                        return config
                    return b''
                v.docker = Mock(side_effect=docker)
                with patch.dict(os.environ, {'BUILDKIT_IMAGE': image}):
                    if mutation is None:
                        v.builder_start()
                    else:
                        with self.assertRaises(RuntimeError):
                            v.builder_start()


class FrozenNodeGate(unittest.TestCase):
    def test_exact_run_and_cleanup_required(self):
        import node_gate
        good = {'status':'completed','conclusion':'success',
                'head_sha':'796134aaa526023dd2c4574bdf807dcaaf67e5b8',
                'head_branch':'ci/validate-node-dfa3138f-20261021','event':'push',
                'path':'.github/workflows/validate-node.yml','run_attempt':1,
                'repository':{'full_name':'Apol-bit/ispeak_backend'}}
        steps = ['Fetch reviewed automation without persistent credentials',
                 'Check validation tooling offline','Validate frozen Node candidate',
                 'Remove only run-owned containers and private temporary files']
        jobs={'jobs':[{'name':'node-candidate','conclusion':'success',
                      'steps':[{'name':n,'conclusion':'success'} for n in steps]}]}
        env={'NODE_VALIDATION_RUN_ID':'39550512234',
             'NODE_AUTOMATION_SHA':good['head_sha']}
        for mode in ('valid','failed','retry','cleanup'):
            run=copy.deepcopy(good); job=copy.deepcopy(jobs)
            if mode=='failed': run['conclusion']='failure'
            if mode=='retry': run['run_attempt']=2
            if mode=='cleanup': job['jobs'][0]['steps'][-1]['conclusion']='failure'
            with self.subTest(mode=mode), patch.dict(os.environ,env), patch.object(node_gate.urllib.request,'urlopen') as urlopen:
                responses=[]
                for payload in (run,job):
                    response=Mock();response.__enter__=Mock(return_value=io.StringIO(json.dumps(payload)));response.__exit__=Mock(return_value=False);responses.append(response)
                urlopen.side_effect=responses
                if mode=='valid':
                    with contextlib.redirect_stdout(io.StringIO()): node_gate.require_node_success()
                else:
                    with self.assertRaises(RuntimeError):node_gate.require_node_success()

    def test_other_success_identity_rejected_without_network(self):
        import node_gate
        with patch.dict(os.environ,{'NODE_VALIDATION_RUN_ID':'123','NODE_AUTOMATION_SHA':'a'*40}), patch.object(node_gate.urllib.request,'urlopen') as get:
            with self.assertRaises(RuntimeError): node_gate.require_node_success()
            get.assert_not_called()


class SyntheticAudioFixture(unittest.TestCase):
    def test_frozen_synthetic_fixture(self):
        import validate_python
        self.assertEqual(len(validate_python.fixture_bytes()),452176)

    def test_missing_and_tampered_fixture_rejected(self):
        import validate_python
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic.wav'
            with patch.object(validate_python,'FIXTURE',path):
                with self.assertRaises(RuntimeError):validate_python.fixture_bytes()
                path.write_bytes(b'not the approved recording')
                with self.assertRaises(RuntimeError):validate_python.fixture_bytes()

    def test_inference_acceptance_unchanged(self):
        import ast
        import validate_python
        source=Path(validate_python.__file__).read_text()
        self.assertNotIn('espeak',source)
        self.assertIn("assert status==200",source)
        self.assertIn("assert status==400 and json.loads(body)['detail']['code']=='recording_too_long'",source)


if __name__ == '__main__':
    unittest.main()
