"""Same requirements.txt version check the old run_benchmark.py ran."""


import importlib.metadata
import re
import sys

from .paths import SCRIPT_DIR

_OPS = {
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    ">":  lambda a, b: a > b,  "<":  lambda a, b: a < b,
}


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[.+]", v) if x.isdigit())


def check_requirements() -> None:
    req_file = SCRIPT_DIR / "requirements.txt"
    missing: list[str] = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*([><=!]{1,2})\s*([0-9][0-9.]*)", line)
        if m:
            pkg, op, req_ver = m.group(1), m.group(2), m.group(3)
        else:
            pkg, op, req_ver = line, None, None
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"  {pkg}: not installed")
            continue
        if op and req_ver and not _OPS[op](_ver(installed), _ver(req_ver)):
            missing.append(f"  {pkg}: need {op}{req_ver}, got {installed}")
    if missing:
        print("Error: missing or outdated requirements:")
        print("\n".join(missing))
        print(f"Install them with: pip install -r {req_file}")
        sys.exit(1)
