# tensor-viterbi

Tensor Hidden Semi-Markov Model (HSMM) Viterbi decoding in Python, C++, and CUDA.

## Provided Functions

| Function | Status | Description |
|---|---|---|
| `decode_vanilla_viterbi` | ✅ Active | Reference O(T·N²·D) triple-loop implementation |
| `decode_log_tensor_viterbi_cached` | ✅ Active | Vectorized log-space tensor implementation with emission caching |
| `decode_tensor_viterbi_cpp` | ✅ Active | C++ tensor implementation (via pybind11) |
| `decode_tensor_viterbi_cuda` | ✅ Active | GPU tensor implementation (via pybind11 + CUDA) |
| `decode_tensor_viterbi_omp` | ✅ Active | OpenMP-parallelized C++ tensor implementation (via pybind11) |
| `decode_log_tensor_viterbi_no_cache` | ⚠️ Deprecated | Log-space tensor without emission caching, slower than cached version |
| `decode_tensor_viterbi` | ⚠️ Deprecated | Linear-space tensor, underflows after ~370 timesteps |


## Actual Issues

- Leonardo: issues on path, if we use default GCC 8.5.0 everything works. If I load GCC 12.2 as a module, it compiles well but it execute using GCC 8.5.0 runtime causing crash.

- Marenostrum:?

- Alps: 

## Repo Structure

```
tensor-viterbi/
├── tensor_viterbi/              # Python package
│   ├── __init__.py
│   ├── hsmm.py                  # HSMM class
│   └── viterbi/
│       ├── tensor.py            # Python tensor implementations
│       ├── vanilla.py
│       ├── native.py            # Python wrappers for C++/CUDA/OMP extensions
│       ├── _native.pyi          # type stubs
│       └── <system>/            # per-system compiled .so (e.g. mi250x/, epyc-7763/)
│           └── _native.so
├── src/                         # C++ / CUDA / ROCm sources
│   ├── bindings.cpp             # pybind11 module
│   ├── hsmm.cu / hsmm.hpp       # CPU and GPU Viterbi implementations
│   └── kernels.cu / kernels.cuh # CUDA/HIP GPU kernels
├── data/                        # JSON model files
├── results/                     # benchmark outputs (gitignored)
│   └── <system>/
│       ├── <Ns>s_<D>d_<T>t.csv
│       ├── <Ns>s_<D>d_<T>t.out
│       └── <Ns>s_<D>d_<T>t.err
├── build/                       # CMake build dirs (gitignored)
│   └── <system>/
├── validation/                  # validation scripts against hsmmlearn
├── hsmmlearn/                   # bundled hsmmlearn (jvkersch) for validation
├── hsmmlearn_omp/               # bundled hsmmlearn with OMP support for validation
├── systems.conf                 # HPC system descriptors (SLURM partitions, modules, GPU arch)
├── run_benchmark.sh             # compile + submit SLURM benchmark sweep
├── run_viterbi.py               # CLI runner (validate / measure / benchmark)
├── compile.sh                   # build native extension for a given system
└── CMakeLists.txt
```

## How to Start

### Requirements

- Python >= 3.10
- numpy
- deprecated


#### (optional) for C++/CUDA native versions:
- CUDA toolkit >= 12.0
- pybind11 >= 2.12
- CMake >= 3.18
- GCC matching your Python environment (see note below)

### Installation

The library is designed to run on HPC clusters. Since some systems require manual module loading while others require specific path configurations for CUDA and GCC, we recommend a two-step setup: first, load the necessary modules (specifically Python, GCC, and CUDA); then, use a virtual environment to manage your pip packages.

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Validation Requirements

Our implementation is validated against [hsmmlearn](https://github.com/jvkersch/hsmmlearn) by jvkersch. Both `hsmmlearn` and `hsmmlearn_omp` are bundled in this repository. To install them:

```bash
pip install ./hsmmlearn
pip install ./hsmmlearn_omp
```

#### Python only (no C++/CUDA)

```bash
pip install numpy
```

The native extension is optional and gracefully skipped if not built:

```python
from tensor_viterbi import HSMM, decode_log_tensor_viterbi_cached
```

#### With C++/CUDA extension (CMake)


**1. Install pybind11**

```bash
pip install pybind11
```

**2. Build with CMake (manual)**

```bash
cmake -B build/<system> -DSYSTEM_NAME=<system> [-DBUILD_GPU=ON/OFF] [-DGPU_PLATFORM=CUDA|ROCM]
cmake --build build/<system> -j 8
```

The `.so` is placed at `tensor_viterbi/viterbi/<system>/_native.so`. Set `SYS_NAME=<system>` in your environment before running so `native.py` loads the correct binary.

**2. Build with `compile.sh` (recommended on HPC)**

```bash
./compile.sh --system <system_name>
```

Loads the correct modules from `systems.conf`, runs CMake via `srun`, and places the `.so` at `tensor_viterbi/viterbi/<system>/_native.so`. Must be run from the repository root.



## How to Run

### As a library

```python
from tensor_viterbi import HSMM, decode_log_tensor_viterbi_cached, decode_vanilla_viterbi

hsmm = HSMM.load_model("data/20states_1000steps_20dur.json")

path = decode_log_tensor_viterbi_cached(hsmm)
```

### Testing

```bash
# Validate against hsmmlearn baseline
python run_viterbi.py -m validate --cpp -dp data/3states_20steps_4dur.json

# Single timing measurement
python run_viterbi.py -m measure --cpp --baseline -dp data/20states_1000steps_20dur.json

# Benchmark (10 iterations per backend, writes CSV to results/<system>/)
python run_viterbi.py -m benchmark --cpp --omp --baseline --system epyc-7763 -dp data/20states_1000steps_20dur.json
```

#### `run_viterbi.py` backend flags

| Flag | Backend |
|---|---|
| `--py` | Python (vectorized, no native ext needed) |
| `--cpp` | C++ (single-threaded, requires native build) |
| `--omp` | C++ with OpenMP parallelism |
| `--cuda` | CUDA / ROCm GPU kernel |
| `--baseline` | HSMMLearn C++ and HSMMLearn-OMP reference implementations |

---

### HPC Benchmarking (`run_benchmark.sh`)

`run_benchmark.sh` compiles the native extension and submits a grid of SLURM batch jobs — one per `(states, duration, timesteps)` combination. Must be run from the repository root.

```bash
./run_benchmark.sh --system <system_name> [backend flags]
```

**Arguments**

| Argument | Required | Description |
|---|---|---|
| `--system <name>` | Yes | System key as defined in `systems.conf` (e.g. `epyc-7763`, `mi250x`, `a100`) |
| `--cpp` | No | Benchmark the C++ backend |
| `--omp` | No | Benchmark the OpenMP backend |
| `--py` | No | Benchmark the Python backend |
| `--cuda` | No | Benchmark the CUDA/ROCm backend |
| `--baseline` | No | Include HSMMLearn C++ and OMP baselines |

> **GPU systems automatically run `--cuda`** regardless of flags. Backend flags are only meaningful for CPU systems.

**What it does**

1. **Compiles** the native extension for the target system via `compile.sh --system <name>`, placing the `.so` at `tensor_viterbi/viterbi/<system>/_native.so`.
2. **Submits SLURM jobs** — one per parameter combination in the `states × durations × timesteps` grid (edit the arrays at the bottom of the script to control the sweep).
3. **Per-job wall-time** is set automatically based on problem size (scales from 1h up to 16h for the largest combinations).
4. **Outputs** are written to `results/<system>/`:
   - `<Ns>s_<D>d_<T>t.out` / `.err` — SLURM stdout/stderr
   - `<Ns>s_<D>d_<T>t.csv` — benchmark timings (function, n_states, timesteps, max_duration, iteration, elapsed_s)

**Examples**

```bash
# CPU node — run C++, OMP, and baseline comparisons
./run_benchmark.sh --system epyc-7763 --cpp --omp --baseline

# GPU node — CUDA is automatic, no flags needed
./run_benchmark.sh --system mi250x

# GPU node, A100 on Leonardo
./run_benchmark.sh --system a100
```

**Adding a new system**

Edit `systems.conf` and add entries to the associative arrays. Required fields by type:

| Field | CPU | GPU |
|---|---|---|
| `SYS_TYPE` | `"cpu"` | `"gpu"` |
| `SYS_PARTITION` | ✅ | ✅ |
| `SYS_ACCOUNT` | ✅ | ✅ |
| `SYS_CPUS` | ✅ | — |
| `SYS_MODULES` | ✅ | ✅ |
| `SYS_MODULES_BUILD` | ✅ | ✅ |
| `SYS_GPU_ARCH` | — | ✅ (SM string for CUDA, GFX target for ROCm) |

### Data format

Models are JSON files with the following fields:

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
