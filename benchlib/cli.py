"""bench — argparse subparsers wiring for plan/run/status/check/likwid."""

import argparse
import sys

from . import compileflow, execution, flags as flagslib, likwid as likwidlib, manifest, params, status
from .paths import require_python_version, require_repo_root
from .requirements import check_requirements
from .systemsconf import SystemConfigError, load_system, select_toolchains


def _add_system_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--system", "-s", required=True,
                    help="System name (systems/<name>.yaml) or a path to a system YAML file")


def _add_toolchain_arg(p: argparse.ArgumentParser, allow_all: bool = True) -> None:
    help_text = "Toolchain key"
    if allow_all:
        help_text += ", or 'all' for every toolchain the system defines"
    help_text += " (defaults to the system's single toolchain if it only has one)"
    p.add_argument("--toolchain", "-t", default=None, help=help_text)


def _add_backend_flag_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--py",           action="store_true")
    p.add_argument("--cpp",          action="store_true")
    p.add_argument("--omp",          action="store_true")
    p.add_argument("--gpu",          action="store_true")
    p.add_argument("--baseline",     action="store_true")
    p.add_argument("--baseline-cpp", action="store_true", dest="baseline_cpp")
    p.add_argument("--baseline-omp", action="store_true", dest="baseline_omp")


def _add_profiler_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--nsys", action="store_true",
                    help="Wrap runs with nsys profile (CUDA timeline tracing)")
    p.add_argument("--ncu", action="store_true",
                    help="Wrap runs with ncu (Nsight Compute kernel profiling; overrides --nsys)")


def _load_or_die(system_arg: str) -> dict:
    try:
        conf, warnings = load_system(system_arg)
    except SystemConfigError as e:
        print(str(e))
        sys.exit(1)
    for w in warnings:
        print(f"Warning in {conf['path']}: {w}")
    return conf


def _resolve_toolchains_or_die(conf: dict, toolchain_arg: str | None, allow_all: bool = True) -> list[str]:
    try:
        return select_toolchains(conf, toolchain_arg, allow_all=allow_all)
    except SystemConfigError as e:
        print(str(e))
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="tensor-viterbi benchmark orchestration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Build the job manifest and print a preview; runs nothing")
    _add_system_arg(p_plan)
    _add_toolchain_arg(p_plan, allow_all=False)
    p_plan.add_argument("--pack", "-p", required=True,
                         help="small|medium|large|extra|stress (stress is GPU-only and always "
                              "uses --gpu)")
    _add_backend_flag_args(p_plan)
    p_plan.add_argument("--iterations", type=int, default=6, metavar="N")

    p_run = sub.add_parser("run", help="Execute a manifest already produced by `bench plan`")
    _add_system_arg(p_run)
    _add_toolchain_arg(p_run, allow_all=False)
    p_run.add_argument("--pack", "-p", default=None,
                        help="Pack to run; if omitted, runs every pack already planned for this system")
    p_run.add_argument("--only-failed", action="store_true",
                        help="Re-run only jobs whose outputs exist but are incomplete/failed")
    p_run.add_argument("--force", action="store_true",
                        help="Re-run jobs even if their output is already complete")
    _add_profiler_args(p_run)

    p_status = sub.add_parser("status", help="Report done/running/pending/failed per job")
    _add_system_arg(p_status)

    p_check = sub.add_parser("check", help="Validate the system YAML and probe the environment; runs nothing")
    _add_system_arg(p_check)
    _add_toolchain_arg(p_check)

    p_likwid = sub.add_parser("likwid", help="LIKWID hardware-counter profiling (CPU-only, fixed data file)")
    _add_system_arg(p_likwid)
    _add_toolchain_arg(p_likwid)
    _add_backend_flag_args(p_likwid)

    p_plot = sub.add_parser("plot", help="Run every plotter in plot/ against results/, saving PNGs")
    p_plot.add_argument("--all-toolchains", action="store_true",
                         help="Include every toolchain found in results/, not just each "
                              "system's default (see DEFAULT_TOOLCHAINS in each plot/*.py)")
    p_plot.add_argument("--system", default=None,
                         help="System key for plot_likwid.py, e.g. xeon8480 (requires --toolchain too)")
    p_plot.add_argument("--toolchain", "-t", default=None,
                         help="Toolchain key for plot_likwid.py, e.g. gnu (requires --system too)")

    return parser


def _plan(args) -> tuple[dict, str, list[dict], list]:
    """Builds and writes the manifest(s) for `bench plan`. Returns (conf, pack, jobs, paths)."""
    conf = _load_or_die(args.system)
    toolchains = _resolve_toolchains_or_die(conf, args.toolchain, allow_all=False)
    multi_toolchain = len(conf["toolchains"]) > 1
    pack = params.resolve_pack_name(args.pack)

    if pack == params.STRESS_PACK:
        if conf["type"] != "gpu":
            print(f"Error: pack 'stress' requires a GPU system "
                  f"(system '{conf['name']}' is type '{conf['type']}').")
            sys.exit(1)
        other_flags = any(getattr(args, f, False) for f in
                           ("py", "cpp", "omp", "baseline", "baseline_cpp", "baseline_omp"))
        if other_flags:
            print("Error: pack 'stress' only supports --gpu; drop the other backend flags.")
            sys.exit(1)
        viterbi_flags = "gpu"
    else:
        viterbi_flags = flagslib.compute_viterbi_flags(args, conf["type"])

        orphans = params.validate_grid_covered_by_packs(viterbi_flags, args.iterations)
        if orphans:
            print("Error: the following grid points have a walltime that falls outside every pack:")
            for s, d, t, wt in orphans:
                print(f"  states={s} duration={d} timesteps={t} walltime={wt}")
            print("Fix walltimes.yaml or the PACKS boundaries in benchlib/params.py.")
            sys.exit(1)

    jobs, skipped = manifest.build_jobs(conf["name"], toolchains, pack, viterbi_flags, args.iterations)

    # Each toolchain gets its own manifest, so preview them separately too —
    # a combined total would overstate what any single `bench run
    # --toolchain <tc>` actually executes (they don't run together).
    paths = []
    if multi_toolchain:
        for i, tc in enumerate(toolchains):
            if i > 0:
                print()
            tc_jobs = [j for j in jobs if j["toolchain"] == tc]
            paths.append(manifest.write_manifest(conf["name"], pack, tc_jobs, toolchain=tc))
            print(f"--- toolchain: {tc} ---")
            manifest.print_preview(conf["name"], pack, tc_jobs, conf["scheduler"])
    else:
        paths.append(manifest.write_manifest(conf["name"], pack, jobs))
        manifest.print_preview(conf["name"], pack, jobs, conf["scheduler"])

    if pack != params.STRESS_PACK:
        print()
        print(f"Pack '{pack}': skipped {skipped} job(s) outside the selected walltime range"
              f"{' (summed across every planned toolchain)' if multi_toolchain else ''}.")
    return conf, pack, jobs, paths


def cmd_plan(args) -> None:
    conf, pack, jobs, paths = _plan(args)
    print()
    for path in paths:
        print(f"Manifest written to {path}")


def cmd_run(args) -> None:
    conf = _load_or_die(args.system)
    scheduler = conf["scheduler"]
    # Errors out if the system defines multiple toolchains and none was
    # picked — same rule `plan`/`check`/`likwid` already enforce, so a
    # multi-toolchain system's several per-toolchain manifests can never be
    # run ambiguously.
    toolchains = _resolve_toolchains_or_die(conf, args.toolchain, allow_all=False)
    multi_toolchain = len(conf["toolchains"]) > 1

    def compile_fn(toolchain: str) -> None:
        compileflow.compile_for(conf, toolchain, scheduler, likwid=False)

    for tc in toolchains:
        tc_arg = tc if multi_toolchain else None
        tag = f"{conf['name']}/{tc}" if multi_toolchain else conf["name"]
        plan_hint = f"--toolchain {tc} " if multi_toolchain else ""

        if args.pack:
            pack_names = [params.resolve_pack_name(args.pack)]
        else:
            tc_dir = manifest.manifest_dir(conf["name"], tc_arg)
            pack_names = sorted(p.stem for p in tc_dir.glob("*.jsonl")) if tc_dir.exists() else []
            if not pack_names:
                print(f"No manifest found for '{tag}'. Pass --pack, or run "
                      f"`bench plan --system {conf['name']} {plan_hint}--pack <pack>` first.")
                sys.exit(1)

        for pack in pack_names:
            path = manifest.manifest_path(conf["name"], pack, tc_arg)
            if not path.exists():
                print(f"No manifest for '{tag}' pack '{pack}'. Run "
                      f"`bench plan --system {conf['name']} {plan_hint}--pack {pack} [flags]` first.")
                sys.exit(1)
            jobs = manifest.read_manifest(path)
            execution.run_manifest(
                jobs, conf, scheduler,
                force=args.force, only_failed=args.only_failed,
                nsys=args.nsys, ncu=args.ncu, compile_fn=compile_fn,
            )


def cmd_status(args) -> None:
    conf = _load_or_die(args.system)
    status.print_status(conf["name"], conf["scheduler"])


def cmd_check(args) -> None:
    from .check import run_check
    sys.exit(run_check(args.system, args.toolchain))


def cmd_likwid(args) -> None:
    conf = _load_or_die(args.system)
    if conf["type"] != "cpu":
        print(f"Warning: LIKWID profiling is CPU-only (system type={conf['type']}). Skipping.")
        return
    toolchains = _resolve_toolchains_or_die(conf, args.toolchain)
    cpu_flags = flagslib.selected_likwid_cpu_flags(args)
    for tc in toolchains:
        print(f"=== Compiling {conf['name']} / {tc} (likwid) ===")
        compileflow.compile_for(conf, tc, conf["scheduler"], likwid=True)
        likwidlib.run_likwid(conf, tc, conf["scheduler"], cpu_flags)


def cmd_plot(args) -> None:
    from . import plotting
    if not plotting.run_all(args.all_toolchains, args.system, args.toolchain):
        sys.exit(1)


def main() -> None:
    require_python_version()
    parser = _build_parser()
    args = parser.parse_args()
    require_repo_root("bench", sys.argv[1:])
    check_requirements()

    {
        "plan":   cmd_plan,
        "run":    cmd_run,
        "status": cmd_status,
        "check":  cmd_check,
        "likwid": cmd_likwid,
        "plot":   cmd_plot,
    }[args.command](args)
