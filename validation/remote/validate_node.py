"""Model-free candidate tests. No Atlas, R2, Python service, or cloud secrets."""
import hashlib
import json
import re
import secrets
import time
from pathlib import Path

from materialize import export_tests


def validate(v, repo, context):
    version = v.docker('run', '--rm', '--network=none', '--memory=256m', '--cpus=1',
                       '--entrypoint=node', v.image, '--version').decode().strip()
    if version != 'v24.21.0':
        raise RuntimeError('Unexpected production Node version')
    tests = export_tests(repo, v.root, v.command)
    # Sibling Python fixtures/development launcher are not in the production image.
    excluded = {'assessmentDetails.test.js', 'fillerAnalysis.test.js', 'localAudioUat.test.js'}
    selected = sorted('/app/test/' + item.name for item in tests.glob('*.test.js') if item.name not in excluded)
    if not selected:
        raise RuntimeError('No production-image tests selected')
    result = v.docker(*v.run_args('node-tests-' + v.key, memory='2g'),
                      '--tmpfs', '/app/uploads:rw,size=64m,mode=1777',
                      '--mount', f'type=bind,src={tests},dst=/app/test,readonly',
                      '--entrypoint=node', v.image, '--test', '--test-concurrency=1', *selected, seconds=240)
    match = re.search(rb'^# pass (\d+)\s*$', result, re.M)
    if not match or int(match[1]) == 0 or re.search(rb'^# fail [1-9]', result, re.M):
        raise RuntimeError('Node image test summary missing or failed')
    v.evidence('node_image_tests', result='PASS', passed=int(match[1]), test_files=len(selected),
               excluded_files=sorted(excluded), full_179_suite_rerun=False)
    status, diagnostic = v.docker(*v.run_args('node-invalid-' + v.key),
                         '--tmpfs', '/app/uploads:rw,size=16m,mode=1777',
                         v.image, seconds=30, check=False)
    if status == 0 or b'Invalid container production configuration' not in diagnostic:
        raise RuntimeError('Missing production configuration unexpectedly accepted')
    fixture = Path(__file__).with_name('node_health_fixture.cjs').resolve()
    settings = {'NODE_ENV': 'production', 'MONGO_URI': 'mongodb://127.0.0.1:1/fixture?tls=true',
                'JWT_SECRET': secrets.token_hex(32), 'PYTHON_SERVICE_TOKEN': secrets.token_hex(32),
                'CORS_ORIGINS': 'https://fixture.example.invalid',
                'PYTHON_BACKEND_URL': 'http://127.0.0.1:8080', 'PORT': '5000',
                'REGISTRATION_MODE': 'open', 'STORAGE_DRIVER': 'local',
                'ALLOW_LOCAL_STORAGE_IN_PRODUCTION': 'true', 'LOCAL_STORAGE_ROOT': '/tmp/uploads',
                'RECORDING_RETENTION_DAYS': '0', 'NODE_OPTIONS': '--require=/fixture.cjs'}
    env = v.root / 'node-fixture.env'
    env.write_text(''.join(key + '=' + value + '\n' for key, value in settings.items()))
    env.chmod(0o600)
    name = 'node-health-' + v.key
    v.docker(*v.run_args(name), '--read-only', '--tmpfs', '/tmp:rw,size=64m,mode=1777',
             '--mount', f'type=bind,src={fixture},dst=/fixture.cjs,readonly',
             '--env-file', str(env), '-d', v.image)
    probe = "fetch('http://127.0.0.1:5000/health').then(async r=>{if(r.status!==200||(await r.json()).database!=='connected')process.exit(1)})"
    for _ in range(25):
        status, _ = v.docker('exec', name, 'node', '-e', probe, seconds=8, check=False)
        if status == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError('Fixture-backed container startup/health failed')
    for _ in range(25):
        state = json.loads(v.docker('inspect', name))[0]['State']
        if state.get('Health', {}).get('Status') == 'healthy':
            break
        time.sleep(1)
    else:
        raise RuntimeError('Docker HEALTHCHECK did not become healthy')
    v.docker('stop', '--time=10', name, seconds=20)
    if json.loads(v.docker('inspect', name))[0]['State']['ExitCode'] != 0:
        raise RuntimeError('Node graceful shutdown failed')
    expected = {item.relative_to(context).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                for item in context.rglob('*') if item.is_file() and item.suffix in ('.js', '.sh')}
    check = """const fs=require('fs'),crypto=require('crypto');
const expected=JSON.parse(process.argv[1]);
for(const [file,hash]of Object.entries(expected)){
 if(crypto.createHash('sha256').update(fs.readFileSync('/app/'+file)).digest('hex')!==hash)process.exit(1);
}
for(const name of ['.env','.git','recordings'])if(fs.existsSync('/app/'+name))process.exit(1);
"""
    v.docker('run', '--rm', '--network=none', '--memory=256m', '--cpus=1', '--entrypoint=node',
             v.image, '-e', check, json.dumps(expected), seconds=30)
    v.evidence('node_startup_security', result='PASS', node_version=version,
               health='fixture-backed HTTP 200 and Docker healthy', graceful_shutdown=True,
               atlas_r2_verified=False, local_storage_exception='isolated fixture only')
