# tensor-viterbi

Tensor Hidden Semi-Markov Model (HSMM) Viterbi decoding implemented in Python, C++, OpenMP, and CUDA/ROCm, with a full benchmarking suite for HPC clusters and local machines.

---

## Implemented Backends

| Function | Status | Description |
|---|---|---|
| `decode_log_tensor_viterbi_cached` | ✅ Active | Vectorized log-space tensor implementation with emission caching (Python) |
| `decode_tensor_viterbi_cpp` | ✅ Active | C++ tensor implementation (via pybind11) |
| `decode_tensor_viterbi_omp` | ✅ Active | OpenMP-parallelized C++ tensor implementation |
| `decode_tensor_viterbi_cuda` | ✅ Active | GPU tensor implementation (CUDA / ROCm via pybind11) |
| `decode_vanilla_viterbi` | ✅ Active | Reference O(T·N²·D) triple-loop implementation |
| `decode_log_tensor_viterbi_no_cache` | ⚠️ Deprecated | Log-space tensor without emission caching |
| `decode_tensor_viterbi` | ⚠️ Deprecated | Linear-space tensor, underflows after ~370 timesteps |

---

## Repository Structure

```
tensor-viterbi/
├── tensor_viterbi/              # Python package
│   ├── __init__.py
│   ├── hsmm.py                  # HSMM class
│   └── viterbi/
│       ├── tensor.py            # Python tensor implementations
│       ├── vanilla.py
│       ├── native.py            # Lazy wrappers for C++/CUDA/OMP extensions
│       ├── _native.pyi          # type stubs
│       └── <system>/<toolchain>/  # per-system compiled .so
│           └── _native.so
├── src/                         # C++ / CUDA / ROCm sources
│   ├── bindings.cpp             # pybind11 module
│   ├── hsmm.cu / hsmm.hpp       # CPU and GPU Viterbi implementations
│   └── kernels.cu / kernels.cuh # CUDA/HIP GPU kernels
├── data/                        # JSON model files
├── results/                     # benchmark outputs (gitignored)
│   └── <system>/<toolchain>/
│       ├── <Ns>s_<D>d_<T>t_<function>.csv
│       ├── <Ns>s_<D>d_<T>t.out
│       └── <Ns>s_<D>d_<T>t.err
├── build/                       # CMake build dirs (gitignored)
│   └── <system>/<toolchain>/
├── .venv/                       # per-toolchain virtual environments (gitignored)
│   └── <system>/<toolchain>/
├── validation/                  # validation scripts against hsmmlearn
├── hsmmlearn/                   # bundled hsmmlearn (CPU baseline)
├── hsmmlearn_omp/               # bundled hsmmlearn with OMP support
├── systems/                     # One YAML file per machine (see TEMPLATE.yaml)
│   └── TEMPLATE.yaml
├── walltimes.yaml               # (states, duration, timesteps) -> estimated walltime
├── benchmark_params.cfg         # The sweep grid (states/durations/timesteps lists)
├── runs/                        # bench plan output: <system>/<pack>.jsonl manifests, or
│                                 #   <system>/<toolchain>/<pack>.jsonl if the system defines
│                                 #   more than one toolchain (gitignored)
├── benchlib/                    # Implementation behind the `bench` CLI
├── bench                        # Entry point: plan / run / status / check / likwid
├── run_one.sh / run_one.slurm   # Single shared job-execution script (local + SLURM)
├── likwid_one.sh / likwid_one.slurm  # Single shared LIKWID profiling script
├── compile.py                   # Library: compiles the native extension (no CLI, imported by bench)
├── viterbi_app.py               # Benchmark executor: runs backends, writes CSVs, validates
├── requirements.txt             # Python dependencies
└── CMakeLists.txt
```

See **[REPRODUCING.md](REPRODUCING.md)** for the full walkthrough (local CPU/GPU,
SLURM CPU/GPU, multi-system reproduction, LIKWID/nsys/ncu). The rest of this
section is a quick reference.

---

## Requirements

- Python >= 3.10
- CMake >= 3.18
- A C++ compiler (GCC, Clang, Intel ICX, Cray CC, Fujitsu FCC)
- For GPU backends: CUDA toolkit >= 12.0 or ROCm

Python packages (see `requirements.txt`):
```
numpy >= 2.4
pandas >= 3.0
matplotlib >= 3.10
scipy >= 1.17
pybind11 >= 3.0
Cython >= 3.2
wheel >= 0.47.0
```

---

## Setup

### 1 — Clone the repository

```bash
git clone https://github.com/lor3ny/tensor-viterbi.git
cd tensor-viterbi
```

### 2 — Describe your system in `systems/<name>.yaml`

There is no auto-detection of scheduler, system, or toolchain anywhere in
this project: `bench` only ever learns about a machine from the YAML file
you point it at with `--system`. Every pre-configured paper system already
has a file under `systems/`; to add your own:

```bash
cp systems/TEMPLATE.yaml systems/my-machine.yaml
# fill in name / type / toolchain / scheduler, delete the slurm: block if local
bench check --system my-machine
```

A minimal local CPU system:

```yaml
name: my-machine
type: cpu
toolchain: gnu
scheduler: local
cpus: 8
```

A minimal SLURM CPU system:

```yaml
name: my-cluster
type: cpu
toolchain: gnu
scheduler: slurm
slurm:
  account: myproject
  partition: normal
  modules: [gcc/12.2]
```

Full schema, optional fields (`omp_bind`, `omp_places`, `metrics_backend`,
`cc`/`cxx`, `gpu_arch`, multi-toolchain systems, etc.) and validation rules
are documented in `systems/TEMPLATE.yaml` and in
**[REPRODUCING.md](REPRODUCING.md)**. `bench check --system <name>` validates
the file and probes the environment (compiler, `sbatch`, `nvcc`/`hipcc`,
`likwid-perfctr`) without running anything.

### 3 — Create and activate a virtual environment

You manage your own virtual environment. Create it, activate it, and install
dependencies before running any script:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

On HPC systems, activate the same environment **before** submitting jobs.
`bench` passes the current `PATH` (including the active venv) to each SLURM
job via `--export=ALL`, so no venv re-activation is needed inside the batch
script.

There is no separate build step. `bench run` (and `bench likwid`) compile the
native extension (via `compile.py`'s `compile_system()`, using CMake and the
currently active Python interpreter) before dispatching jobs, every time
they're invoked. `compile.py` has no CLI of its own — it cannot be invoked
standalone to "only compile".

`bench` must be run from the repository root.

---

## Running Benchmarks

| Command | Role |
|---|---|
| `bench plan` | Builds the job manifest (`runs/<system>/<pack>.jsonl`) and prints a preview; runs nothing |
| `bench run` | Executes the manifest — sbatch for SLURM systems, direct call for local systems; resumes by default |
| `bench status` | Reports done/running/pending/failed per job |
| `bench check` | Validates the system YAML and probes the environment; runs nothing |
| `bench likwid` | LIKWID hardware-counter profiling (CPU only, fixed data file; see [known incompatibilities](REPRODUCING.md#6-likwid-and-nsysncu-profiling) for unsupported AMD CPUs) |
| `bench plot` | Runs every plotter in `plot/` against `results/`, saving PNGs |
| `viterbi_app.py` | Executes one benchmark: runs backends, writes CSVs, validates results |

### Running the benchmark grid

```bash
bench plan --system <system> --pack <pack> [backend flags]
bench run  --system <system> [--pack <pack>]
bench status --system <system>
```

`bench run` plans implicitly if no manifest exists yet for the given pack, so
for a one-shot run `bench run --system <system> --pack <pack> [backend flags]`
is enough. The scheduler (`local` or `slurm`) is never a CLI flag — it comes
from `systems/<system>.yaml`.

If a system defines more than one toolchain (e.g. `epyc-7763-bigmem`, which
has `cray`/`aocc`/`gnu`), each toolchain gets its own manifest —
`runs/<system>/<toolchain>/<pack>.jsonl` — so planning `gnu` doesn't overwrite
`cray`'s plan for the same pack. `bench run` then requires `--toolchain <tc>`
on such systems, for the same reason `bench plan` already does: there's no
single manifest to fall back to. Single-toolchain systems are unaffected —
their manifests stay at the flat `runs/<system>/<pack>.jsonl`.

**Backend flags** (CPU systems — pick one or more; GPU runs `--gpu` automatically):

| Flag | Backend |
|---|---|
| `--cpp` | C++ single-threaded |
| `--omp` | C++ OpenMP |
| `--py` | Python vectorized |
| `--gpu` | CUDA / ROCm (GPU only) |
| `--baseline` | HSMMLearn C++ + OMP reference |
| `--baseline-cpp` | HSMMLearn C++ only |
| `--baseline-omp` | HSMMLearn OMP only |

**Other `bench run` flags:**

| Flag | Default | Description |
|---|---|---|
| `--toolchain <tc>` | system's only toolchain | Which toolchain to run; **required** if the system defines more than one (each toolchain has its own manifest — see below) |
| `--iterations N` | 6 | Benchmark repetitions per job (capped at 2 for T ≥ 1M); only used if planning implicitly |
| `--only-failed` | off | Re-run only jobs whose output is incomplete/failed |
| `--force` | off | Re-run jobs even if already complete |
| `--nsys` / `--ncu` | off | Wrap runs with Nsight Systems / Nsight Compute (`--ncu` wins if both given) |

Examples:
```bash
# SLURM — CPU node, C++, OpenMP and baselines
bench run --system xeon8480 --toolchain intel --pack medium --cpp --omp --baseline

# SLURM — GPU node (--gpu selected automatically)
bench run --system a100 --pack small

# SLURM — one toolchain of a multi-toolchain node
bench run --system epyc-7763-bigmem --toolchain gnu --pack large --cpp --omp

# Local machine
bench run --system workstation --pack small --cpp --omp

# The expensive jobs
bench run --system xeon8480 --toolchain intel --pack extra --cpp --omp
```

### Walltime packs

The full `states × durations × timesteps` grid spans a wide range of estimated
walltimes (`walltimes.yaml`). `--pack` on `bench plan` (and optionally
`bench run`) lets an evaluator pick how much wall-clock budget they want to
spend without editing `benchmark_params.cfg`. Jobs outside the selected
bucket are skipped and a skip count is printed. Buckets follow the natural
gaps in the walltime table (nothing falls between 8h and 10h):

| Pack | Walltime range | Jobs in default grid |
|---|---|---|
| `small` | ≤ 1 hour | 36 |
| `medium` | 1–2 hours | 6 |
| `large` | 2–8 hours | 11 |
| `extra` | 8–20 hours | 7 |

There is no way to submit the full unfiltered grid in one invocation — run
each pack separately if you want to cover everything. See
[REPRODUCING.md](REPRODUCING.md) for per-job walltimes and slicing flags for
serial local runs.

There's also a `stress` pack: it isn't a walltime bucket over
`benchmark_params.cfg` like the ones above, it's a dedicated single-point
grid (`benchmark_params_stress.cfg`, `states=100`/`durations=10000`/
`timesteps=10000000`) for GPU-only stress testing. `bench plan --pack stress`
requires a GPU system and always runs `--gpu` — passing any other backend
flag alongside it is an error.

### Running a single file directly

`viterbi_app.py` can also be called directly to benchmark one data file without
going through the sweep. This is useful for quick checks on a login node.
It does **not** compile anything — run `bench run` (or `bench likwid`) at
least once for the target system/toolchain first so
`tensor_viterbi/viterbi/<system>/<toolchain>/_native.so` exists.

```bash
python viterbi_app.py --system <system> --toolchain <toolchain> \
    --cpp --omp --baseline --iterations 3 \
    --data-path data/10states_1000steps_100dur.json
```

Validation against the reference (saved as `<data>_reference.npy`) runs
automatically on the last iteration of each backend.

---

## Customising the Parameter Sweep

The grid of jobs planned by `bench plan` is controlled by
`benchmark_params.cfg` in the repository root:

```
states    = 10, 15, 25, 50, 75
durations = 100, 250, 500, 1000
timesteps = 10000, 100000, 1000000
```

One job is submitted for every combination. Edit these lists before running to
change the sweep:

```
# Add an intermediate timestep
timesteps = 100000, 1000000

# Drop the smaller states
states = 50, 75
```

`timesteps=10000000` is deliberately not in this grid — it only runs through
the dedicated `stress` pack (`benchmark_params_stress.cfg`, `--pack stress`,
GPU-only), not by editing this file.

Use `--pack` (see above) to run a size-bounded subset of this grid without
editing the config file.

Each combination requires a pre-generated model file at
`data/<N>states_<T>steps_<D>dur.json`. If you add new parameter values,
generate the corresponding files first:

```bash
python data/data_generator.py
```

---

## Reproducing the Results

See **[REPRODUCING.md](REPRODUCING.md)** for the full, copy-pasteable
walkthrough: setup, describing your system, the universal
plan/run/status loop, local CPU/GPU and SLURM CPU/GPU scenarios,
multi-system reproduction (run per machine, `rsync` `results/` together),
the pack ladder, and LIKWID/nsys/ncu profiling.

Quick version:

```bash
git clone https://github.com/lor3ny/tensor-viterbi.git && cd tensor-viterbi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp systems/TEMPLATE.yaml systems/my-machine.yaml   # fill it in
bench check --system my-machine

bench plan --system my-machine --pack small --cpp --omp
bench run  --system my-machine
bench status --system my-machine
```

Results land in `results/<system>/<toolchain>/<Ns>s_<D>d_<T>t_<function>.csv`
(columns: `function, n_states, timesteps, max_duration, iteration, elapsed_s`).

---

## Output Format

Results land in `results/<system>/<toolchain>/`. For each job:

| File | Description |
|---|---|
| `<Ns>s_<D>d_<T>t_<function>.csv` | Timing rows, one per iteration |
| `<Ns>s_<D>d_<T>t_<function>_metrics.csv` | Energy/power metrics (if collector configured) |
| `<Ns>s_<D>d_<T>t.out` | stdout (hardware diagnostics + benchmark output) |
| `<Ns>s_<D>d_<T>t.err` | stderr |

CSV columns: `function, n_states, timesteps, max_duration, iteration, elapsed_s`

---

## Data Format

Model files are JSON with the following fields:

```json
{
  "n_steps": 1000,
  "M": 20,
  "n_bins": 13,
  "seed": 42,
  "pi": [...],
  "trans_mat": [[...]],
  "obs_seq": [...],
  "states": [
    { "name": "S0", "emission_probs": [...], "duration_probs": [...] }
  ]
}
```

Generate new data files with:

```bash
python data/data_generator.py
```

---

## Known Issues

- **Leonardo (CINECA)**: mixing GCC versions can cause runtime crashes. Using the
  default GCC 8.5.0 (no explicit compiler module) is stable; loading GCC 12.2 compiles
  but links against the wrong runtime.
- **GPU venvs**: built with `--system-site-packages` and only install `numpy` and
  `pybind11` directly. All other packages (`pandas`, `scipy`, etc.) must be available
  via the system Python module loaded in the system's `systems/<name>.yaml`.
