"""Hosted Linux resource gates; thresholds cannot be overridden by environment."""
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

GIB = 1024 ** 3
START_RAM = {'node': 5 * GIB, 'python': 8 * GIB}
EMERGENCY_RAM = 5 * GIB
START_DOCKER_FREE = {'node': 20 * GIB, 'python': 50 * GIB}
CONTINUOUS_DOCKER_FREE = {'node': 15 * GIB, 'python': 50 * GIB}
SYSTEM_FREE = 15 * GIB


def filesystem(path):
    stats = os.statvfs(path)
    if stats.f_flag & os.ST_RDONLY or stats.f_favail <= 0:
        raise RuntimeError('Backing filesystem is read-only or out of inodes')
    return stats.f_bavail * stats.f_frsize, stats.f_blocks * stats.f_frsize


def measure():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    info = json.loads(subprocess.check_output(
        ['docker', 'info', '--format', '{{json .}}'], timeout=15, stderr=subprocess.DEVNULL))
    if info.get('OSType') != 'linux' or info.get('Architecture') not in ('x86_64', 'amd64'):
        raise RuntimeError('Linux amd64 Docker engine required')
    if os.environ.get('DOCKER_HOST') not in (None, '', 'unix:///var/run/docker.sock'):
        raise RuntimeError('Unexpected Docker endpoint')
    context = json.loads(subprocess.check_output(['docker', 'context', 'inspect'],
                                                timeout=15, stderr=subprocess.DEVNULL))[0]
    if context['Endpoints']['docker']['Host'] != 'unix:///var/run/docker.sock':
        raise RuntimeError('Standard hosted-runner local Docker endpoint required')
    root = Path(info['DockerRootDir']).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError('Docker backing path not established')
    config_path = Path('/etc/docker/daemon.json')
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get('data-root') and Path(config['data-root']).resolve() != root:
            raise RuntimeError('Conflicting Docker storage mapping')
        if config.get('containerd') not in (None, '/run/containerd/containerd.sock', '/var/run/containerd/containerd.sock'):
            raise RuntimeError('Unreviewed external containerd store')
    containerd = Path('/var/lib/containerd')
    containerd_config = Path('/etc/containerd/config.toml')
    if containerd_config.exists():
        containerd = Path(tomllib.loads(containerd_config.read_text()).get('root', str(containerd)))
    stores = [root]
    if containerd.is_dir():
        stores.append(containerd.resolve(strict=True))
    elif any('containerd' in str(row) for row in info.get('DriverStatus', [])):
        raise RuntimeError('Containerd backing path not established')
    free, capacity = zip(*(filesystem(path) for path in stores))
    return {'available_ram_bytes': int(values['MemAvailable'].split()[0]) * 1024,
            'docker_memory_bytes': info['MemTotal'], 'docker_free_bytes': min(free),
            'docker_capacity_bytes': min(capacity), 'system_free_bytes': filesystem('/')[0]}


def problems(role, data, *, start):
    ram = START_RAM[role] if start else EMERGENCY_RAM
    disk = START_DOCKER_FREE[role] if start else CONTINUOUS_DOCKER_FREE[role]
    return [name for name, failed in (
        ('RAM', data['available_ram_bytes'] < ram or data['docker_memory_bytes'] < ram),
        ('DOCKER_DISK', data['docker_free_bytes'] < disk),
        ('SYSTEM_DISK', data['system_free_bytes'] < SYSTEM_FREE)) if failed]


def enforce(role, *, stable=False):
    minimum = None
    for sample in range(13 if stable else 1):
        data = measure()
        failed = problems(role, data, start=stable)
        print(json.dumps({'check': 'resources', 'role': role, 'sample': sample,
                          'measurements': data, 'failed_gates': failed}), flush=True)
        if failed:
            raise RuntimeError('Resource gate failed; no retry')
        minimum = data if minimum is None else {key: min(data[key], minimum[key]) for key in data}
        if stable and sample != 12:
            time.sleep(5)
    return minimum
