# tensor-viterbi

Tensor Hidden Semi-Markov Model (HSMM) Viterbi decoding implemented in Python, C++, OpenMP, and CUDA/ROCm, with a full benchmarking suite for HPC clusters and local machines.

---

## Implemented Backends

| Function | Status | Description |
|---|---|---|
| `decode_log_tensor_viterbi_cached` | ✅ Active | Vectorized log-space tensor implementation with emission caching (Python) |
| `decode_tensor_viterbi_cpp` | ✅ Active | C++ tensor implementation (via pybind11) |
| `decode_tensor_viterbi_omp` | ✅ Active | OpenMP-parallelized C++ tensor implementation |
| `decode_tensor_viterbi_omp_opt` | ✅ Active | Optimized OpenMP C++ tensor implementation |
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
├── systems.json                 # System descriptors (scheduler, partitions, modules, GPU arch)
├── compile.py                   # Build native extension for a given system/toolchain
├── run_benchmark.py             # Job submitter: sbatch (SLURM) or direct (local)
├── viterbi_app.py               # Benchmark executor: runs backends, writes CSVs, validates
├── requirements.txt             # Python dependencies
└── CMakeLists.txt
```

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
```

---

## Setup

### 1 — Clone the repository

```bash
git clone https://github.com/lor3ny/tensor-viterbi.git
cd tensor-viterbi
```

### 2 — Configure your system in `systems.json`

Every system (laptop, HPC node, cloud VM) must have an entry in `systems.json`.
The `"scheduler"` field controls how `run_benchmark.py` dispatches jobs:

| Value | Behaviour |
|---|---|
| `"local"` | Calls `python viterbi_app.py` directly in the current shell |
| `"slurm"` | Generates a `.slrm` script and submits it via `sbatch` |

A minimal local CPU entry:

```json
"workstation": {
  "scheduler": "local",
  "type": "cpu",
  "cpus": 8,
  "toolchains": {
    "gnu": {
      "modules": "",
      "modules_build": ""
    }
  }
}
```

For SLURM clusters add `partition`, `account`, and the module names to load:

```json
"my-cluster-cpu": {
  "scheduler": "slurm",
  "type": "cpu",
  "partition": "compute",
  "account": "myproject",
  "cpus": 128,
  "toolchains": {
    "gnu": {
      "modules": "Python/3.11:GCC/12",
      "modules_build": "Python/3.11:GCC/12"
    }
  }
}
```

For GPU nodes add `gpu_arch` (CUDA SM string or ROCm GFX target) instead of `cpus`:

```json
"my-cluster-gpu": {
  "scheduler": "slurm",
  "type": "gpu",
  "partition": "gpu",
  "account": "myproject",
  "gpu_arch": "80",
  "toolchains": {
    "cuda": {
      "modules": "CUDA/12:Python/3.11",
      "modules_build": "CUDA/12:Python/3.11"
    }
  }
}
```

All pre-configured systems are already in `systems.json`. Run `python compile.py --help`
to print the list of available systems and toolchains.

**Optional fields** (add inside the system or toolchain dict only when needed):

| Field | Scope | Description |
|---|---|---|
| `qos` | system | SLURM QOS string (`--qos`) |
| `uenv` | toolchain | uenv image name (Alps/CSCS only) |
| `metrics_backend` | toolchain | Energy/power metrics collector token |
| `omp_bind` | system | `OMP_PROC_BIND` override |
| `omp_places` | system | `OMP_PLACES` override |
| `cc` / `cxx` | toolchain | Explicit compiler path override |

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
`run_benchmark.py` passes the current `PATH` (including the active venv) to
each SLURM job via `--export=ALL`, so no venv re-activation is needed inside
the batch script.

### 4 — Build the native extension

`compile.py` verifies requirements and compiles the C++/CUDA extension via CMake
using the Python interpreter that is currently active. It does not create or
modify any virtual environment.

```bash
python compile.py --system <system> --toolchain <toolchain>
```

Examples:
```bash
# Workstation (no modules)
python compile.py --system workstation --toolchain gnu

# Intel Xeon on Leonardo (CINECA)
python compile.py --system xeon8480 --toolchain intel

# A100 GPU on Leonardo (CINECA)
python compile.py --system a100 --toolchain cuda

# Build all toolchains for a system at once
python compile.py --system epyc-7763-bigmem --toolchain all
```

Both `compile.py` and `run_benchmark.py` must be run from the repository root.

---

## Running Benchmarks

The benchmark workflow is split across two scripts:

| Script | Role |
|---|---|
| `run_benchmark.py` | Dispatches jobs — sbatch for SLURM systems, direct call for local systems |
| `viterbi_app.py` | Executes one benchmark: runs backends, writes CSVs, validates results |


### Running the benchmark grid

```bash
python run_benchmark.py --system <system> --toolchain <toolchain> [backend flags]
```

**Backend flags** (CPU systems — pick one or more; GPU runs `--cuda` automatically):

| Flag | Backend |
|---|---|
| `--cpp` | C++ single-threaded |
| `--omp` | C++ OpenMP |
| `--omp-opt` | C++ OpenMP optimized |
| `--py` | Python vectorized |
| `--cuda` | CUDA / ROCm (GPU only) |
| `--baseline` | HSMMLearn C++ + OMP reference |
| `--baseline-cpp` | HSMMLearn C++ only |
| `--baseline-omp` | HSMMLearn OMP only |

**Other flags:**

| Flag | Default | Description |
|---|---|---|
| `--iterations N` | 6 | Benchmark repetitions per job (capped at 2 for T ≥ 1M) |
| `--toolchain all` | — | Run every toolchain defined for the system |

Examples:
```bash
# SLURM — CPU node, C++, OpenMP and baselines
python run_benchmark.py --system xeon8480 --toolchain intel --cpp --omp --baseline

# SLURM — GPU node (CUDA selected automatically)
python run_benchmark.py --system a100 --toolchain cuda

# SLURM — all toolchains for a node
python run_benchmark.py --system epyc-7763-bigmem --toolchain all --cpp --omp

# Local machine (scheduler: local in systems.json)
python run_benchmark.py --system workstation --toolchain gnu --cpp --omp
```

### Running a single file directly

`viterbi_app.py` can also be called directly to benchmark one data file without
going through the sweep. This is useful for quick checks on a login node.

```bash
python viterbi_app.py --system <system> --toolchain <toolchain> \
    --cpp --omp --baseline --iterations 3 \
    --data-path data/10states_1000steps_100dur.json
```

Validation against the reference (saved as `<data>_reference.npy`) runs
automatically on the last iteration of each backend.

---

## Customising the Parameter Sweep

The grid of jobs submitted by `run_benchmark.py` is controlled by three lists
near the top of that file:

```python
STATES    = [10, 15, 25, 50, 75]
DURATIONS = [100, 250, 500, 1000]
TIMESTEPS = [1000000]
```

One job is submitted for every combination. Edit these lists before running to
change the sweep:

```python
# Single large configuration
STATES    = [100]
DURATIONS = [10000]
TIMESTEPS = [10000000]

# Add an intermediate timestep
TIMESTEPS = [100000, 1000000]
```

Each combination requires a pre-generated model file at
`data/<N>states_<T>steps_<D>dur.json`. If you add new parameter values,
generate the corresponding files first:

```bash
python data/data_generator.py
```

---

## Reproducing the Results

The full sequence to reproduce the benchmark results from scratch on an HPC system:

**HPC cluster (SLURM):**

```bash
# 1. Clone and enter the repo
git clone https://github.com/lor3ny/tensor-viterbi.git
cd tensor-viterbi

# 2. Create and activate your environment, install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build — run once per system/toolchain pair
#    (also compiles hsmmlearn baselines with the correct toolchain compiler)
python compile.py --system <system> --toolchain <toolchain>

# 4. Submit the full benchmark grid (venv must still be active)
#    scheduler: slurm in systems.json → generates .tmp_benchmark.slrm and sbatches it
python run_benchmark.py --system <system> --toolchain <toolchain>

# 5. Results land in results/<system>/<toolchain>/
#    <Ns>s_<D>d_<T>t_<function>.csv
#    Columns: function, n_states, timesteps, max_duration, iteration, elapsed_s
```

**Local machine / workstation:**

```bash
# 1. Add your system to systems.json with "scheduler": "local"
#    (see "Configure your system" above)

# 2. Create and activate your environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build (also compiles hsmmlearn baselines)
python compile.py --system workstation --toolchain gnu

# 4. Run the benchmark grid
#    scheduler: local in systems.json → calls viterbi_app.py directly
python run_benchmark.py --system workstation --toolchain gnu --cpp --omp

# 5. Quick single-file check
python viterbi_app.py --system workstation --toolchain gnu \
    --cpp --omp --iterations 3 -dp data/10states_1000steps_100dur.json
```

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
  via the system Python module loaded in `systems.json`.
