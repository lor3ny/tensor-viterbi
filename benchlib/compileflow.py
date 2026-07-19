"""Adapter from the new systems/*.yaml config shape to compile.py's public
`compile_system(system, toolchain, sys_conf, tc_conf, scheduler, likwid)`
interface, which is left untouched.
"""

from compile import compile_system


def compile_for(conf: dict, toolchain: str, scheduler: str, likwid: bool = False) -> None:
    sys_conf: dict = {"type": conf["type"]}
    if conf.get("gpu_arch"):
        sys_conf["gpu_arch"] = conf["gpu_arch"]
    slurm = conf["slurm"]
    if slurm.get("account"):
        sys_conf["account"] = slurm["account"]
    if slurm.get("partition"):
        sys_conf["partition"] = slurm["partition"]
    if slurm.get("qos"):
        sys_conf["qos"] = slurm["qos"]

    tc_conf = conf["toolchains"][toolchain]
    compile_system(conf["name"], toolchain, sys_conf, tc_conf, scheduler, likwid)
