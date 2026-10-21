"""Publish only an already validated local image; never build or deploy."""
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ACCEPT = 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'


def get(url, headers=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'iSpeak-release', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read(2 * 1024 * 1024), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, b'', {}
    except Exception:
        raise RuntimeError('Registry verification unavailable; no retry') from None


def private_package(name, credential, absent_ok=False):
    status, body, _ = get('https://api.github.com/user/packages/container/' + name,
                          {'Authorization': 'Bearer ' + credential, 'Accept': 'application/vnd.github+json'})
    if status == 404 and absent_ok:
        return  # New GHCR packages default to private; verify after first push.
    if status != 200 or json.loads(body).get('visibility') != 'private':
        raise RuntimeError('Private GHCR visibility not established')


def publish(v, image):
    if os.environ.get('PUBLICATION_APPROVED') != 'true':
        raise RuntimeError('Separate publication approval required')
    credential = os.environ.get('GHCR_READ_TOKEN', '')
    if not credential or os.environ.get('GITHUB_ACTOR') != 'Apol-bit':
        raise RuntimeError('Protected Publisher credential required')
    name = 'ispeak-' + v.role
    package = 'apol-bit/' + name
    tag = 'testing-' + image['Config']['Labels']['org.opencontainers.image.revision'][:12] + '-' + v.key
    reference = 'ghcr.io/' + package + ':' + tag
    private_package(name, credential, absent_ok=v.role == 'node')
    basic = base64.b64encode(('Apol-bit:' + credential).encode()).decode()
    url = 'https://ghcr.io/token?' + urllib.parse.urlencode({'service':'ghcr.io','scope':'repository:' + package + ':pull'})
    status, body, _ = get(url, {'Authorization': 'Basic ' + basic})
    if status != 200:
        raise RuntimeError('Publisher package read authorization failed')
    bearer = json.loads(body)['token']
    headers = {'Authorization':'Bearer ' + bearer, 'Accept':ACCEPT}
    manifest_url = 'https://ghcr.io/v2/' + package + '/manifests/'
    status, _, _ = get(manifest_url + tag, headers)
    if status != 404:
        raise RuntimeError('Unique release tag absence not established')
    config = Path(v.docker_env['DOCKER_CONFIG'])
    config.mkdir(mode=0o700, exist_ok=True)
    v.command(['docker','login','ghcr.io','-u','Apol-bit','--password-stdin'], input_data=(credential+'\n').encode())
    if json.loads(v.docker('image','inspect',v.image))[0]['Id'] != image['Id']:
        raise RuntimeError('Validated local image identity changed')
    v.docker('tag',v.image,reference)
    immutable = None
    try:
        v.publication_started = True
        v.evidence('publication_attempt', reference=reference, tested_image_id=image['Id'])
        output = v.docker('push',reference,seconds=600)
        digests = re.findall(rb'digest: (sha256:[a-f0-9]{64})',output)
        if len(set(digests)) != 1:
            raise RuntimeError('Push digest unavailable; remote publication may exist')
        digest = digests[0].decode()
        v.published_digest = digest
        v.evidence('image_pushed',reference=reference,digest=digest, tested_image_id=image['Id'])
        status, body, _ = get(manifest_url + digest,headers)
        if status != 200 or 'sha256:'+hashlib.sha256(body).hexdigest() != digest or json.loads(body).get('config',{}).get('digest') != image['Id']:
            raise RuntimeError('Published manifest does not match tested image')
        private_package(name,credential)
        # Test unauthenticated manifest access, including an anonymous bearer.
        status, _, _ = get(manifest_url+digest,{'Accept':ACCEPT})
        if status not in (401,403,404):
            raise RuntimeError('Anonymous manifest access not denied')
        status, body, _ = get(url)
        if status == 200:
            anonymous=json.loads(body)['token']
            status, _, _ = get(manifest_url+digest,{'Authorization':'Bearer '+anonymous,'Accept':ACCEPT})
        if status not in (401,403,404):
            raise RuntimeError('Anonymous bearer access not denied')
        immutable='ghcr.io/'+package+'@'+digest
        v.docker('pull','--platform=linux/amd64',immutable,seconds=300)
        if json.loads(v.docker('image','inspect',immutable))[0]['Id'] != image['Id']:
            raise RuntimeError('Authenticated digest pull identity mismatch')
        v.evidence('publication_verified',reference=reference,immutable=immutable,image_id=image['Id'],private=True,anonymous_denied=True,published=True,railway_reader_verified=False)
    finally:
        if immutable:
            v.docker('image','rm',immutable,seconds=20,check=False)
        v.docker('image','rm',reference,seconds=20,check=False)
