import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import publish_image as pub


class Publication(unittest.TestCase):
    def fixture(self, temp):
        image={'Id':'sha256:'+'a'*64,'Config':{'Labels':{'org.opencontainers.image.revision':'d'*40}}}
        manifest=json.dumps({'config':{'digest':image['Id']}}).encode()
        digest='sha256:'+hashlib.sha256(manifest).hexdigest()
        v=Mock();v.role='node';v.key='123-1';v.image='local-candidate';v.docker_env={'DOCKER_CONFIG':str(Path(temp)/'auth')}
        def docker(*args,**kw):
            if args[:2]==('image','inspect'):return json.dumps([image]).encode()
            if args[0]=='push':return ('digest: '+digest).encode()
            return b''
        v.docker.side_effect=docker
        def get(url,headers=None):
            headers=headers or {}
            if 'api.github.com' in url:return 200,b'{"visibility":"private"}',{}
            if '/token?' in url:
                if headers.get('Authorization','').startswith('Basic '):return 200,b'{"token":"authorized-fixture"}',{}
                return 200,b'{"token":"anonymous-fixture"}',{}
            if '/manifests/testing-' in url:return 404,b'',{}
            if headers.get('Authorization')=='Bearer authorized-fixture':return 200,manifest,{}
            return 401,b'',{}
        return v,image,get

    def test_disabled_has_no_external_effects(self):
        with patch.dict(os.environ,{'PUBLICATION_APPROVED':'false'}),patch.object(pub,'get') as get:
            with self.assertRaises(RuntimeError):pub.publish(Mock(),{})
            get.assert_not_called()

    def test_publish_exact_tested_image_and_digest_pull(self):
        with tempfile.TemporaryDirectory() as temp:
            v,image,get=self.fixture(temp)
            with patch.dict(os.environ,{'PUBLICATION_APPROVED':'true','GHCR_READ_TOKEN':'synthetic-token','GITHUB_ACTOR':'Apol-bit'}),patch.object(pub,'get',side_effect=get):pub.publish(v,image)
            commands=[c.args for c in v.docker.call_args_list]
            self.assertEqual(sum(c[0]=='push' for c in commands),1)
            self.assertTrue(any(c[0]=='pull' and '@sha256:' in c[-1] for c in commands))
            self.assertFalse(any(c[0]=='buildx' for c in commands))
            self.assertNotIn('synthetic-token',str(commands))
            self.assertTrue(any(c.kwargs.get('published') is True for c in v.evidence.call_args_list))

    def test_visibility_collision_and_manifest_fail_closed(self):
        for mode in ['public','collision','manifest','anonymous']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                v,image,good=self.fixture(temp)
                def get(url,headers=None):
                    if mode=='public' and 'api.github.com' in url:return 200,b'{"visibility":"public"}',{}
                    if mode=='collision' and '/manifests/testing-' in url:return 200,b'{}',{}
                    if mode=='manifest' and '/manifests/sha256:' in url:return 200,b'{}',{}
                    if mode=='anonymous' and '/manifests/sha256:' in url and not (headers or {}).get('Authorization'):return 200,b'{}',{}
                    return good(url,headers)
                with patch.dict(os.environ,{'PUBLICATION_APPROVED':'true','GHCR_READ_TOKEN':'synthetic-token','GITHUB_ACTOR':'Apol-bit'}),patch.object(pub,'get',side_effect=get):
                    with self.assertRaises(RuntimeError):pub.publish(v,image)
                if mode in ['public','collision']:self.assertFalse(any(c.args[0]=='push' for c in v.docker.call_args_list))

    def test_missing_credential_stops(self):
        with patch.dict(os.environ,{'PUBLICATION_APPROVED':'true','GHCR_READ_TOKEN':'','GITHUB_ACTOR':'Apol-bit'}):
            with self.assertRaises(RuntimeError):pub.publish(Mock(),{})


if __name__=='__main__':unittest.main()
