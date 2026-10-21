"""Require a reviewed successful Node workflow run, not an operator-written PASS flag."""
import json
import os
import re
import urllib.request


def require_node_success():
    run_id, sha = os.environ.get('NODE_VALIDATION_RUN_ID', ''), os.environ.get('NODE_AUTOMATION_SHA', '')
    if not run_id.isdigit() or not re.fullmatch('[a-f0-9]{40}', sha):
        raise RuntimeError('Reviewed Node run identity has not been supplied')
    base = f'https://api.github.com/repos/Apol-bit/ispeak_backend/actions/runs/{run_id}'

    def get(url):
        # Public run metadata: no Python repository token sent to another repo.
        request = urllib.request.Request(url, headers={'User-Agent': 'iSpeak-release-validation',
                                                     'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    run = get(base)
    expected = {'status': 'completed', 'conclusion': 'success', 'head_sha': sha,
                'head_branch': 'ci/validate-node-dfa3138f-20261021', 'event': 'push',
                'path': '.github/workflows/validate-node.yml', 'run_attempt': 1}
    if any(run.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Node run does not match reviewed successful automation')
    if run.get('repository', {}).get('full_name') != 'Apol-bit/ispeak_backend':
        raise RuntimeError('Unexpected Node validation repository')
    jobs = get(base + '/attempts/1/jobs?per_page=100')['jobs']
    matches = [job for job in jobs if job['name'] == 'node-candidate' and job['conclusion'] == 'success']
    if len(matches) != 1:
        raise RuntimeError('Successful Node candidate job not found')
    required = ['Fetch reviewed automation without persistent credentials', 'Validate frozen Node candidate']
    for name in required:
        if not any(step['name'] == name and step['conclusion'] == 'success' for step in matches[0]['steps']):
            raise RuntimeError('Required Node validation step did not pass')
    print(json.dumps({'node_gate': 'PASS', 'run_id': run_id, 'reviewed_automation_sha': sha}), flush=True)
