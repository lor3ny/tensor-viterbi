"""Cross-invocation guard for `bench run` on scheduler: local systems.

A local run executes directly on the invoking machine's CPU/GPU (see
run_one.sh), so a second `bench run` for a different system or toolchain
started on the same machine while the first is still going would silently
share those resources and corrupt both runs' timings. This keeps a single
lock file recording who's currently running; a conflicting invocation is
flagged and aborted instead of racing ahead.

Not used for scheduler: slurm — those jobs execute on an exclusively
allocated compute node (see run_one.slurm's --exclusive), not on the
machine that invoked `bench run`, so there's no shared-resource conflict
to guard against.
"""

import atexit
import json
import os
import socket
import sys
import time

from .paths import RUNS_DIR

LOCK_PATH = RUNS_DIR / ".bench_run.lock"

_owned = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock() -> dict | None:
    try:
        return json.loads(LOCK_PATH.read_text())
    except (OSError, ValueError):
        return None


def acquire(system: str, toolchain: str) -> None:
    global _owned
    hostname = socket.gethostname()
    held = _read_lock()
    if held and held.get("hostname") == hostname and _pid_alive(held.get("pid", -1)):
        if held["system"] != system or held["toolchain"] != toolchain:
            print(f"Error: another `bench run` is active on this machine "
                  f"({hostname}): system={held['system']} toolchain={held['toolchain']} "
                  f"(pid={held['pid']}, started {held['started']}).")
            print(f"Refusing to start system={system} toolchain={toolchain} here — "
                  f"local runs execute directly on this machine's CPU/GPU, so running "
                  f"another architecture/toolchain at the same time would corrupt both "
                  f"runs' timings. Wait for it to finish, or stop it if it's stale.")
        else:
            print(f"Error: system={system} toolchain={toolchain} is already running on "
                  f"this machine (pid={held['pid']}, started {held['started']}).")
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps({
        "pid": os.getpid(),
        "hostname": hostname,
        "system": system,
        "toolchain": toolchain,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }))
    _owned = True
    atexit.register(release)


def release() -> None:
    global _owned
    if not _owned:
        return
    held = _read_lock()
    if held and held.get("pid") == os.getpid():
        LOCK_PATH.unlink(missing_ok=True)
    _owned = False
