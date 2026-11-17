"""Read-only host audit. Run separately on Ariel login and allocated GPU nodes."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except Exception as exc:
        return {"error": str(exc)}


def collect(paths=()):
    packages = {p.metadata['Name']: p.version for p in importlib.metadata.distributions()}
    result = {
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable, "cwd": os.getcwd(),
        "disk_bytes": dict(zip(('total', 'used', 'free'), shutil.disk_usage('.'))),
        "packages": dict(sorted(packages.items())),
        "commands": {" ".join(c): command(c) for c in [
            ['nvidia-smi'], ['nvcc', '--version'],
            ['sinfo', '-o', '%P %a %l %D %G'], ['scontrol', 'show', 'partition'],
            ['conda', 'info', '--envs'], ['df', '-h', '.']]},
        "paths": {str(p): {"exists": Path(p).exists(),
                           "resolved": str(Path(p).resolve())} for p in paths},
    }
    try:
        import torch
        result['torch'] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "devices": [{"name": torch.cuda.get_device_properties(i).name,
                                        "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                                       for i in range(torch.cuda.device_count())]}
    except Exception as exc:
        result['torch'] = {"error": str(exc)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--path', action='append', default=[])
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(collect(args.path), f, indent=2)
    print(output)
