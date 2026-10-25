"""Model-free checks for the reviewed security dependencies and their consumers."""
import importlib.metadata
from pathlib import Path
import ssl
import tempfile
import unittest

import certifi
from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout


class DependencySecurityTests(unittest.TestCase):
    def test_exact_security_versions(self):
        for package, version in [('certifi', '2026.10.19'), ('filelock', '4.2.1'),
                                 ('setuptools', '84.1.0')]:
            with self.subTest(package=package):
                self.assertEqual(importlib.metadata.version(package), version)

    def test_trust_bundle_builds_ssl_context_without_network(self):
        context = ssl.create_default_context(cafile=certifi.where())
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertGreater(context.cert_store_stats()['x509_ca'], 0)

    def test_filelock_nested_contention_timeout_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'test.lock')
            lock = FileLock(path)
            self.assertIsInstance(lock, BaseFileLock)
            with lock:
                with lock:
                    self.assertEqual(lock.lock_counter, 2)
                    with self.assertRaises(Timeout):
                        FileLock(path).acquire(timeout=0)
                self.assertTrue(lock.is_locked)
            self.assertFalse(lock.is_locked)
            with FileLock(path, timeout=0):
                pass

    def test_soft_filelock_release_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'soft.lock')
            with SoftFileLock(path):
                with self.assertRaises(Timeout):
                    SoftFileLock(path).acquire(timeout=0)
            with SoftFileLock(path, timeout=0):
                pass

    def test_huggingface_lock_without_download_or_model(self):
        from huggingface_hub.utils._fixes import WeakFileLock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hub.lock'
            with WeakFileLock(path, timeout=1):
                with self.assertRaises(Timeout):
                    FileLock(path).acquire(timeout=0)
            with WeakFileLock(path, timeout=1):
                pass

    def test_torch_filelock_subclass_without_model(self):
        from torch.utils._filelock import FileLock as TorchFileLock
        with tempfile.TemporaryDirectory() as directory:
            lock = TorchFileLock(str(Path(directory) / 'torch.lock'))
            with lock:
                self.assertTrue(lock.is_locked)
            self.assertFalse(lock.is_locked)

    def test_setuptools_interfaces_without_compilation(self):
        from setuptools import Extension, distutils
        from setuptools.command.build_ext import build_ext
        from setuptools.dist import Distribution
        self.assertEqual(Extension('synthetic', sources=[]).name, 'synthetic')
        self.assertIsNotNone(distutils)
        self.assertIsNotNone(build_ext)
        self.assertIsInstance(Distribution(), Distribution)


if __name__ == '__main__':
    unittest.main()
