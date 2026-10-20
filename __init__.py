"""Top-level package for XTDDFT."""

from pathlib import Path as _Path
import subprocess as _subprocess
import json as _json
import os as _os
import platform as _platform


def _git_version():
    root = _Path(__file__).resolve().parent
    git = ["git", f"--git-dir={root / '.git'}", f"--work-tree={root}"]
    try:
        information = _subprocess.check_output(
            git + ["log", "-1", "--format=medium", "--no-abbrev-commit",
                   "--date=default", "--no-decorate", "--no-color", "--no-notes"],
            cwd=root, stderr=_subprocess.DEVNULL, text=True, timeout=2,
        ).rstrip()
        status = _subprocess.run(
            git + ["diff", "--quiet", "HEAD"], cwd=root,
            stderr=_subprocess.DEVNULL, timeout=2,
        )
        if status.returncode == 1:
            information += "\n\nWorking tree: dirty (uncommitted changes)"
        return information
    except (OSError, _subprocess.SubprocessError):
        return "Git version: unknown"


def _command(*args):
    try:
        return _subprocess.check_output(
            args, stderr=_subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
    except (OSError, _subprocess.SubprocessError, UnicodeError):
        return ""


def _device_information():
    system = _platform.system()
    cpu = _platform.processor() or "unknown"
    if system == "Linux":
        try:
            models = [line.split(":", 1)[1].strip()
                      for line in _Path("/proc/cpuinfo").read_text().splitlines()
                      if line.split(":", 1)[0].strip() in ("model name", "Hardware")]
            cpu = "; ".join(dict.fromkeys(models)) or cpu
        except (OSError, UnicodeError):
            pass

    memory = "unknown"
    try:
        import psutil
        total = psutil.virtual_memory().total
        memory = f"{total / 1024**3:.2f} GiB"
    except (ImportError, OSError, AttributeError):
        try:
            total = _os.sysconf("SC_PHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
            if total > 0:
                memory = f"{total / 1024**3:.2f} GiB"
        except (OSError, ValueError, AttributeError):
            pass

    gpu = _command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    if system == "Darwin":
        try:
            displays = _json.loads(_command(
                "system_profiler", "SPDisplaysDataType", "SPHardwareDataType", "-json"))
            for hardware in displays.get("SPHardwareDataType", []):
                cpu = hardware.get("chip_type") or hardware.get("cpu_type") or cpu
            gpu = "; ".join(device["sppci_model"]
                            for device in displays.get("SPDisplaysDataType", [])
                            if device.get("sppci_model"))
        except (ValueError, TypeError, KeyError):
            pass
    gpu = "; ".join(gpu.splitlines()) or "unknown (unavailable or not detected)"
    return f"CPU: {cpu}\nGPU (detected): {gpu}\nSystem memory (total): {memory}"


print(" XTDDFT version information ".center(75, "="))
print(_git_version())
print(" current equipment information ".center(75, "="))
print(_device_information())
print('=' * 75)

from . import XTDDFT, utils

__all__ = ["XTDDFT", "utils"]
