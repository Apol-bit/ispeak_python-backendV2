"""Linux-only resource measurements; no thresholds can be supplied by environment."""
import json
import shutil
import subprocess
import time
from pathlib import Path

GIB = 1024 ** 3
START_RAM = {'node': 5 * GIB, 'python': 8 * GIB}
EMERGENCY_RAM = 5 * GIB
DOCKER_FREE = 50 * GIB
SYSTEM_FREE = 15 * GIB


def measure():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    info = json.loads(subprocess.check_output(['docker', 'info', '--format', '{{json .}}'], timeout=15))
    if info.get('OSType') != 'linux' or info.get('Architecture') not in ('x86_64', 'amd64'):
        raise RuntimeError('Linux amd64 Docker engine required')
    root = Path(info['DockerRootDir'])
    # Fail rather than substituting workspace disk for an inaccessible daemon disk.
    if not root.is_dir():
        raise RuntimeError('Cannot establish actual Docker backing-filesystem space')
    stores = [root]
    # Current Docker installations may keep image content in containerd storage,
    # separate from DockerRootDir. Include that filesystem when present.
    containerd = Path('/var/lib/containerd')
    if containerd.is_dir():
        stores.append(containerd)
    if any('containerd' in str(row) for row in info.get('DriverStatus', [])) and not containerd.is_dir():
        raise RuntimeError('Containerd image-store backing path not established')
    return {'available_ram_bytes': available, 'docker_free_bytes': min(shutil.disk_usage(path).free for path in stores),
            'system_free_bytes': shutil.disk_usage('/').free, 'docker_memory_bytes': info['MemTotal']}


def enforce(role, *, stable=False):
    minimum = None
    for sample in range(13 if stable else 1):
        data = measure()
        floor = START_RAM[role] if stable else EMERGENCY_RAM
        if data['available_ram_bytes'] < floor or data['docker_memory_bytes'] < floor:
            raise RuntimeError('RAM gate failed')
        if data['docker_free_bytes'] < DOCKER_FREE or data['system_free_bytes'] < SYSTEM_FREE:
            raise RuntimeError('Disk-space gate failed')
        minimum = data if minimum is None else {key: min(data[key], minimum[key]) for key in data}
        if stable and sample != 12:
            time.sleep(5)
    print(json.dumps({'resource_minima': minimum, 'stable_seconds': 60 if stable else 0}), flush=True)
    return minimum
