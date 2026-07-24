"""Lightweight tests for the bench orchestration layer.

No real benchmarks or sbatch calls are made: dispatch tests stub
subprocess.run. Run with: pytest tests/
"""

from __future__ import annotations

import json

import pytest
import yaml

from benchlib import execution, flags as flagslib, manifest as manifestlib, params, status
from benchlib.systemsconf import SystemConfigError, load_system, select_toolchains


# ── per-version walltime table ─────────────────────────────────────────

def test_grid_fully_covered_by_packs():
    orphans = params.validate_grid_covered_by_packs("cpp:omp:baseline-cpp:baseline-omp", 6)
    assert orphans == []


def test_get_walltime_sums_requested_versions():
    # Large enough grid point that neither term is swallowed by MIN_WALLTIME_SECONDS.
    solo_cpp = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "cpp", 1))
    solo_omp = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "omp", 1))
    combined = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "cpp:omp", 1))
    assert combined == solo_cpp + solo_omp


def test_get_walltime_baseline_flag_expands_to_both_backends():
    baseline = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "baseline", 1))
    split = params.hms_to_seconds(
        params.get_walltime(75, 1000, 100000, "baseline-cpp:baseline-omp", 1)
    )
    assert baseline == split


def test_get_walltime_scales_with_iterations():
    one = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "cpp", 1))
    six = params.hms_to_seconds(params.get_walltime(75, 1000, 100000, "cpp", 6))
    assert six == min(one * 6, params.MAX_WALLTIME_SECONDS)


def test_get_walltime_is_clamped_to_min_and_max():
    tiny = params.get_walltime(10, 100, 10000, "cpp", 1)
    assert params.hms_to_seconds(tiny) >= params.MIN_WALLTIME_SECONDS

    huge = params.get_walltime(75, 1000, 1000000, "cpp:omp:baseline-cpp:baseline-omp", 1000)
    assert params.hms_to_seconds(huge) == params.MAX_WALLTIME_SECONDS


def test_get_walltime_unmeasured_version_falls_back_to_default():
    assert params.get_walltime(10, 100, 10000, "py", 1) == params.seconds_to_hms(
        params._DEFAULT_SECONDS
    )


# ── pack bucketing ──────────────────────────────────────────────────────

def test_pack_boundaries_match_old_values():
    assert params.PACKS == {
        "small":  (0, 3600),
        "medium": (3601, 7200),
        "large":  (7201, 28800),
        "extra":  (28801, 72000),
    }


@pytest.mark.parametrize("old", ["1h", "2h", "4-8h", "10-20h"])
def test_old_pack_names_are_no_longer_accepted(old):
    with pytest.raises(SystemExit):
        params.resolve_pack_name(old)


@pytest.mark.parametrize("walltime,expected_pack", [
    ("00:30:00", "small"), ("01:00:00", "small"),
    ("02:00:00", "medium"),
    ("04:00:00", "large"), ("08:00:00", "large"),
    ("10:00:00", "extra"), ("20:00:00", "extra"),
])
def test_pack_of_walltime(walltime, expected_pack):
    assert params.pack_of_walltime(walltime) == expected_pack


# ── stress pack ─────────────────────────────────────────────────────────

def test_stress_pack_resolves_and_is_not_a_walltime_bucket():
    assert params.resolve_pack_name("stress") == params.STRESS_PACK
    assert params.STRESS_PACK not in params.PACKS


def test_stress_params_file_is_the_single_large_configuration():
    states, durations, timesteps = params.load_benchmark_params(params.STRESS_PARAMS_FILE)
    assert states == [100]
    assert durations == [10000]
    assert timesteps == [10000000]


def test_manifest_build_jobs_stress_pack_ignores_walltime_bucketing():
    jobs, skipped = manifestlib.build_jobs("dummy", ["cuda"], "stress", "gpu", 6)
    assert skipped == 0
    assert len(jobs) == 1
    job = jobs[0]
    assert (job["states"], job["duration"], job["timesteps"]) == (100, 10000, 10000000)
    assert job["flags"] == "gpu"
    assert job["iterations"] == 2  # capped: timesteps >= 1_000_000


# ── manifest generation ────────────────────────────────────────────────

def test_manifest_job_count_per_pack_sums_to_full_grid():
    states, durations, timesteps = params.load_benchmark_params()
    total = len(states) * len(durations) * len(timesteps)
    seen_stems = set()
    grand_total = 0
    for pack in params.PACKS:
        jobs, skipped = manifestlib.build_jobs("dummy", ["gnu"], pack, "cpp:omp", 6)
        grand_total += len(jobs)
        assert len(jobs) + skipped == total
        for j in jobs:
            key = (j["states"], j["duration"], j["timesteps"])
            assert key not in seen_stems, f"{key} assigned to more than one pack"
            seen_stems.add(key)
    assert grand_total == total


def test_manifest_caps_iterations_for_large_timesteps():
    jobs, _ = manifestlib.build_jobs("dummy", ["gnu"], "extra", "cpp", 6)
    for j in jobs:
        if j["timesteps"] >= 1_000_000:
            assert j["iterations"] == 2
        else:
            assert j["iterations"] == 6


# ── per-toolchain manifest nesting ──────────────────────────────────────

def test_manifest_path_flat_without_toolchain():
    path = manifestlib.manifest_path("sysA", "small")
    assert path == manifestlib.RUNS_DIR / "sysA" / "small.jsonl"


def test_manifest_path_nests_under_toolchain_when_given():
    path = manifestlib.manifest_path("sysA", "small", toolchain="gnu")
    assert path == manifestlib.RUNS_DIR / "sysA" / "gnu" / "small.jsonl"


def test_write_manifest_per_toolchain_does_not_clobber_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(manifestlib, "RUNS_DIR", tmp_path)

    gnu_jobs, _ = manifestlib.build_jobs("sysA", ["gnu"], "small", "cpp", 6)
    cray_jobs, _ = manifestlib.build_jobs("sysA", ["cray"], "small", "cpp", 6)
    manifestlib.write_manifest("sysA", "small", gnu_jobs, toolchain="gnu")
    manifestlib.write_manifest("sysA", "small", cray_jobs, toolchain="cray")

    assert manifestlib.read_manifest(tmp_path / "sysA" / "gnu" / "small.jsonl") == gnu_jobs
    assert manifestlib.read_manifest(tmp_path / "sysA" / "cray" / "small.jsonl") == cray_jobs


def test_select_toolchains_requires_explicit_choice_for_multi_toolchain_system(tmp_path):
    path = _write_system(tmp_path, {
        "name": "multi", "type": "cpu", "scheduler": "local",
        "toolchains": {"gnu": {}, "cray": {}},
    })
    conf, _ = load_system(str(path))
    with pytest.raises(SystemConfigError, match="multiple toolchains"):
        select_toolchains(conf, None)
    assert select_toolchains(conf, "gnu") == ["gnu"]
    assert sorted(select_toolchains(conf, "all")) == ["cray", "gnu"]


def test_status_all_manifests_finds_nested_toolchain_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "RUNS_DIR", tmp_path)

    gnu_jobs, _ = manifestlib.build_jobs("sysA", ["gnu"], "small", "cpp", 6)
    cray_jobs, _ = manifestlib.build_jobs("sysA", ["cray"], "small", "cpp", 6)
    (tmp_path / "sysA" / "gnu").mkdir(parents=True)
    (tmp_path / "sysA" / "cray").mkdir(parents=True)
    with open(tmp_path / "sysA" / "gnu" / "small.jsonl", "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in gnu_jobs)
    with open(tmp_path / "sysA" / "cray" / "small.jsonl", "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in cray_jobs)

    found = dict(status.all_manifests("sysA"))
    assert set(found) == {"gnu/small", "cray/small"}
    assert found["gnu/small"] == gnu_jobs
    assert found["cray/small"] == cray_jobs


def test_manifest_multi_toolchain_doubles_jobs():
    jobs_one, _ = manifestlib.build_jobs("dummy", ["gnu"], "small", "cpp", 6)
    jobs_two, _ = manifestlib.build_jobs("dummy", ["gnu", "intel"], "small", "cpp", 6)
    assert len(jobs_two) == 2 * len(jobs_one)
    assert {j["toolchain"] for j in jobs_two} == {"gnu", "intel"}


def test_write_and_read_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(manifestlib, "RUNS_DIR", tmp_path)
    jobs, _ = manifestlib.build_jobs("sysA", ["gnu"], "small", "cpp", 6)
    path = manifestlib.write_manifest("sysA", "small", jobs)
    assert path == tmp_path / "sysA" / "small.jsonl"
    round_tripped = manifestlib.read_manifest(path)
    assert round_tripped == jobs
    # one JSON object per line
    assert len(path.read_text().splitlines()) == len(jobs)
    for line in path.read_text().splitlines():
        json.loads(line)  # must not raise


# ── backend flag selection ─────────────────────────────────────────────

class _Args:
    py = cpp = omp = gpu = baseline = baseline_cpp = baseline_omp = False


def test_default_flags_cpu_vs_gpu():
    assert flagslib.compute_viterbi_flags(_Args(), "cpu") == "cpp:omp:baseline-cpp:baseline-omp"
    assert flagslib.compute_viterbi_flags(_Args(), "gpu") == "gpu"


def test_explicit_flags_override_default():
    args = _Args()
    args.cpp = True
    assert flagslib.compute_viterbi_flags(args, "cpu") == "cpp"


def test_expected_csv_stems_expands_baseline():
    assert set(flagslib.expected_csv_stems("baseline")) == {"HSMMLearn_CPP", "HSMMLearn_OMP"}
    assert flagslib.expected_csv_stems("baseline-cpp") == ["HSMMLearn_CPP"]
    assert set(flagslib.expected_csv_stems("cpp:omp")) == {
        "decode_tensor_viterbi_cpp", "decode_tensor_viterbi_omp",
    }


# ── systems/*.yaml validation ───────────────────────────────────────────

def _write_system(tmp_path, content: dict):
    path = tmp_path / "sys.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


def test_valid_local_system_loads_with_defaults(tmp_path):
    path = _write_system(tmp_path, {
        "name": "my-machine", "type": "cpu", "toolchain": "gnu", "scheduler": "local",
    })
    conf, warnings = load_system(str(path))
    assert warnings == []
    assert conf["omp_bind"] == "close"
    assert conf["omp_places"] == "cores"
    assert conf["cpus"] > 0
    assert list(conf["toolchains"]) == ["gnu"]


def test_missing_required_fields_error(tmp_path):
    path = _write_system(tmp_path, {"name": "x"})
    with pytest.raises(SystemConfigError):
        load_system(str(path))


def test_gpu_only_toolchain_requires_gpu_type(tmp_path):
    path = _write_system(tmp_path, {
        "name": "bad", "type": "cpu", "toolchain": "cuda", "scheduler": "local",
    })
    with pytest.raises(SystemConfigError, match="GPU"):
        load_system(str(path))


def test_gpu_type_requires_gpu_arch(tmp_path):
    path = _write_system(tmp_path, {
        "name": "bad", "type": "gpu", "toolchain": "cuda", "scheduler": "local",
    })
    with pytest.raises(SystemConfigError, match="gpu_arch"):
        load_system(str(path))


def test_slurm_scheduler_requires_account(tmp_path):
    path = _write_system(tmp_path, {
        "name": "bad", "type": "cpu", "toolchain": "gnu", "scheduler": "slurm",
        "slurm": {"partition": "normal"},
    })
    with pytest.raises(SystemConfigError, match="account"):
        load_system(str(path))


def test_slurm_qos_only_is_valid_without_partition(tmp_path):
    """Mirrors h100: no partition, only qos — must not raise, and must not
    even warn since qos satisfies the partition-or-qos requirement."""
    path = _write_system(tmp_path, {
        "name": "h100-like", "type": "gpu", "gpu_arch": "90", "toolchain": "cuda",
        "scheduler": "slurm", "slurm": {"account": "acct1", "qos": "myqos"},
    })
    conf, warnings = load_system(str(path))
    assert conf["slurm"]["account"] == "acct1"
    assert conf["slurm"]["partition"] is None
    assert warnings == []


def test_slurm_missing_both_partition_and_qos_warns(tmp_path):
    path = _write_system(tmp_path, {
        "name": "x", "type": "cpu", "toolchain": "gnu",
        "scheduler": "slurm", "slurm": {"account": "acct1"},
    })
    conf, warnings = load_system(str(path))
    assert any("partition" in w and "qos" in w for w in warnings)


def test_unknown_key_warns_not_errors(tmp_path):
    path = _write_system(tmp_path, {
        "name": "x", "type": "cpu", "toolchain": "gnu", "scheduler": "local",
        "made_up_field": 123,
    })
    conf, warnings = load_system(str(path))
    assert any("made_up_field" in w for w in warnings)


def test_modules_list_normalized_to_colon_string(tmp_path):
    path = _write_system(tmp_path, {
        "name": "x", "type": "cpu", "toolchain": "gnu", "scheduler": "slurm",
        "slurm": {"account": "a1", "partition": "p1", "modules": ["gcc/12.2", "cmake"]},
    })
    conf, _ = load_system(str(path))
    assert conf["toolchains"]["gnu"]["modules"] == "gcc/12.2:cmake"


def test_multi_toolchain_map_with_diverging_build_modules(tmp_path):
    path = _write_system(tmp_path, {
        "name": "x", "type": "gpu", "gpu_arch": "gfx90a", "scheduler": "slurm",
        "toolchains": {
            "amd": {"modules": ["a", "b"], "modules_build": ["a", "rocm", "b"]},
        },
        "slurm": {"account": "a1", "partition": "p1"},
    })
    conf, _ = load_system(str(path))
    assert conf["toolchains"]["amd"]["modules"] == "a:b"
    assert conf["toolchains"]["amd"]["modules_build"] == "a:rocm:b"


def test_file_not_found_lists_available_systems():
    with pytest.raises(SystemConfigError, match="Available systems"):
        load_system("this-system-does-not-exist")


# ── resume / completeness check ─────────────────────────────────────────

def _job(flags="cpp", iterations=6):
    return {"stem": "10s_100d_10000t", "flags": flags, "iterations": iterations}


def test_pending_when_no_output_at_all(tmp_path):
    status, _ = execution.check_job_complete(_job(), tmp_path)
    assert status == "pending"


def _write_csv(dir_path, fname, iterations):
    p = dir_path / f"10s_100d_10000t_{fname}.csv"
    lines = ["function,n_states,timesteps,max_duration,iteration,elapsed_s"]
    lines += [f"{fname},10,10000,100,{i},0.01" for i in range(iterations)]
    p.write_text("\n".join(lines) + "\n")


def test_done_when_all_csvs_complete(tmp_path):
    (tmp_path / "10s_100d_10000t.out").write_text("ok\n")
    (tmp_path / "10s_100d_10000t.err").write_text("")
    _write_csv(tmp_path, "decode_tensor_viterbi_cpp", 6)
    status, detail = execution.check_job_complete(_job(flags="cpp", iterations=6), tmp_path)
    assert status == "done", detail


def test_failed_when_csv_missing(tmp_path):
    (tmp_path / "10s_100d_10000t.out").write_text("ok\n")
    status, detail = execution.check_job_complete(_job(flags="cpp"), tmp_path)
    assert status == "failed"
    assert "missing" in detail


def test_failed_when_csv_truncated(tmp_path):
    (tmp_path / "10s_100d_10000t.out").write_text("ok\n")
    _write_csv(tmp_path, "decode_tensor_viterbi_cpp", 2)  # wanted 6
    status, detail = execution.check_job_complete(_job(flags="cpp", iterations=6), tmp_path)
    assert status == "failed"
    assert "truncated" in detail


def test_failed_on_traceback_even_if_csvs_complete(tmp_path):
    (tmp_path / "10s_100d_10000t.out").write_text("ok\n")
    (tmp_path / "10s_100d_10000t.err").write_text("Traceback (most recent call last):\nboom\n")
    _write_csv(tmp_path, "decode_tensor_viterbi_cpp", 6)
    status, detail = execution.check_job_complete(_job(flags="cpp", iterations=6), tmp_path)
    assert status == "failed"
    assert "traceback" in detail


# ── sbatch flags / job env (pure functions, no subprocess) ─────────────

def _conf(**overrides):
    base = {
        "name": "sysA", "type": "cpu", "cpus": 16,
        "omp_bind": "close", "omp_places": "cores", "gpu_arch": "",
        "toolchains": {"gnu": {"modules": "m1:m2", "uenv": "", "metrics_backend": ""}},
        "slurm": {"account": "acct1", "partition": "part1", "qos": None},
    }
    base.update(overrides)
    return base


def test_build_sbatch_flags_cpu():
    flags = execution.build_sbatch_flags(_conf())
    assert "--account=acct1" in flags
    assert "--partition=part1" in flags
    assert "--cpus-per-task=16" in flags
    assert not any(f.startswith("--qos") for f in flags)


def test_build_sbatch_flags_gpu_uses_gres():
    flags = execution.build_sbatch_flags(_conf(type="gpu"))
    assert "--gres=gpu:1" in flags
    assert not any(f.startswith("--cpus-per-task") for f in flags)


def test_build_job_env_resolves_absolute_data_path():
    conf = _conf()
    job = {"flags": "cpp", "iterations": 6, "stem": "x", "data_path": "data/foo.json"}
    env = execution.build_job_env(conf, "gnu", job, execution.results_dir_for("sysA", "gnu"), False, False)
    assert env["DATA_PATH"].endswith("/data/foo.json")
    assert env["DATA_PATH"].startswith("/")
    assert env["SYS_NAME"] == "sysA/gnu"
    assert env["SYS_MODULES"] == "m1:m2"
    assert env["VITERBI_FLAGS"] == "cpp"


# ── dispatch: stub subprocess.run, never actually run/submit ───────────

def test_dispatch_slurm_stubbed(monkeypatch, tmp_path):
    calls = []

    class FakeResult:
        returncode = 0
        stdout = "Submitted batch job 123"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    monkeypatch.setattr(execution.time, "sleep", lambda *_: None)

    job = {"stem": "x", "flags": "cpp", "iterations": 6, "walltime": "00:30:00",
           "toolchain": "gnu", "data_path": "data/foo.json"}
    execution.dispatch_slurm(job, _conf(), tmp_path, ["--account=acct1"], nsys=False, ncu=False)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "sbatch"
    assert "--account=acct1" in cmd
    assert any(c.startswith("--time=00:30:00") for c in cmd)
    assert any(c.startswith("--job-name=tv_x") for c in cmd)


def test_dispatch_local_stubbed(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    job = {"stem": "x", "flags": "cpp", "iterations": 6, "toolchain": "gnu",
           "data_path": "data/foo.json"}
    execution.dispatch_local(job, _conf(), tmp_path, nsys=False, ncu=False)

    assert (tmp_path / "x.out").exists()
    assert (tmp_path / "x.err").exists()
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == "bash"
    assert kwargs["env"]["SYS_NAME"] == "sysA/gnu"


# ── run_manifest: resume/skip, --force, --only-failed ──

def _fake_jobs(n):
    return [
        {"stem": f"job{i}", "toolchain": "gnu", "flags": "cpp", "iterations": 6,
         "states": i, "duration": 1, "timesteps": 10000, "walltime": "00:30:00"}
        for i in range(n)
    ]


def test_run_manifest_skips_done_reruns_failed_by_default(monkeypatch, tmp_path):
    jobs = _fake_jobs(3)
    monkeypatch.setattr(execution, "results_dir_for", lambda system, tc: tmp_path)

    # job0: done (complete CSV); job1: failed (missing CSV); job2: pending (no output)
    (tmp_path / "job0.out").write_text("ok\n")
    _write_csv_named(tmp_path, "job0", "decode_tensor_viterbi_cpp", 6)
    (tmp_path / "job1.out").write_text("ok\n")  # CSV missing -> failed

    dispatched = []
    monkeypatch.setattr(execution, "dispatch_local",
                         lambda job, conf, results_dir, nsys, ncu: dispatched.append(job["stem"]))

    execution.run_manifest(jobs, _conf(), "local", force=False, only_failed=False,
                            nsys=False, ncu=False,
                            compile_fn=lambda tc: None)

    assert dispatched == ["job1", "job2"]  # job0 skipped (done), job1+job2 run


def test_run_manifest_only_failed_skips_pending(monkeypatch, tmp_path):
    jobs = _fake_jobs(2)
    monkeypatch.setattr(execution, "results_dir_for", lambda system, tc: tmp_path)
    (tmp_path / "job0.out").write_text("ok\n")  # failed: CSV missing
    # job1: pending, no output at all

    dispatched = []
    monkeypatch.setattr(execution, "dispatch_local",
                         lambda job, conf, results_dir, nsys, ncu: dispatched.append(job["stem"]))

    execution.run_manifest(jobs, _conf(), "local", force=False, only_failed=True,
                            nsys=False, ncu=False,
                            compile_fn=lambda tc: None)

    assert dispatched == ["job0"]


def test_run_manifest_force_reruns_done(monkeypatch, tmp_path):
    jobs = _fake_jobs(1)
    monkeypatch.setattr(execution, "results_dir_for", lambda system, tc: tmp_path)
    (tmp_path / "job0.out").write_text("ok\n")
    _write_csv_named(tmp_path, "job0", "decode_tensor_viterbi_cpp", 6)

    dispatched = []
    monkeypatch.setattr(execution, "dispatch_local",
                         lambda job, conf, results_dir, nsys, ncu: dispatched.append(job["stem"]))

    execution.run_manifest(jobs, _conf(), "local", force=True, only_failed=False,
                            nsys=False, ncu=False,
                            compile_fn=lambda tc: None)

    assert dispatched == ["job0"]


def test_run_manifest_compiles_once_per_toolchain(monkeypatch, tmp_path):
    jobs = _fake_jobs(3)
    monkeypatch.setattr(execution, "results_dir_for", lambda system, tc: tmp_path)
    monkeypatch.setattr(execution, "dispatch_local", lambda *a, **k: None)
    compiled = []

    execution.run_manifest(jobs, _conf(), "local", force=False, only_failed=False,
                            nsys=False, ncu=False,
                            compile_fn=lambda tc: compiled.append(tc))

    assert compiled == ["gnu"]  # all 3 jobs share toolchain "gnu" -> compiled once


def _write_csv_named(dir_path, stem, fname, iterations):
    p = dir_path / f"{stem}_{fname}.csv"
    lines = ["function,n_states,timesteps,max_duration,iteration,elapsed_s"]
    lines += [f"{fname},1,10000,1,{i},0.01" for i in range(iterations)]
    p.write_text("\n".join(lines) + "\n")
