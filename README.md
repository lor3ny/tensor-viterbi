# tensor-viterbi

Tensor Hidden Semi-Markov Model (HSMM) Viterbi decoding in Python, C++, and CUDA.

## Provided Functions

| Function | Status | Description |
|---|---|---|
| `decode_vanilla_viterbi` | ✅ Active | Reference O(T·N²·D) triple-loop implementation |
| `decode_log_tensor_viterbi_cached` | ✅ Active | Vectorized log-space tensor implementation with emission caching |
| `decode_tensor_viterbi_cpp` | ✅ Active | C++ tensor implementation (via pybind11) |
| `decode_tensor_viterbi_cuda` | ✅ Active | GPU tensor implementation (via pybind11 + CUDA) |
| `decode_log_tensor_viterbi_no_cache` | ⚠️ Deprecated | Log-space tensor without emission caching, slower than cached version |
| `decode_tensor_viterbi` | ⚠️ Deprecated | Linear-space tensor, underflows after ~370 timesteps |


## Actual Issues

- Leonardo: issues on path, if we use default GCC 8.5.0 everything works. If I load GCC 12.2 as a module, it compiles well but it execute using GCC 8.5.0 runtime causing crash.

- Marenostrum:?

- Alps: 

## Repo Structure

```
tensor-viterbi/
├── tensor_viterbi/          # Python package
│   ├── __init__.py          
│   ├── hsmm.py              # HSMM class
│   └── viterbi/
│       ├── tensor.py        # Python tensor implementations
│       ├── vanilla.py       
│       └── _native.pyi      # type stubs for C++/CUDA extension
├── src/
│   ├── bindings.cpp         
│   └── src/                 # C++ / CUDA sources
│       ├── hsmm.cu / .hpp
│       ├── kernels.cu / .cuh
│       └── main.cpp
├── data/                    # JSON model files
├── validation/              # hsmmlearn baseline
├── run_viterbi.py           # CLI runner
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

Oour implementation is validated against the hsmmlearn repository by jvkersch, if you want to use our validation scripts you need to clone the repo in the main folder. Just use the following command:

```bash
git clone https://github.com/jvkersch/hsmmlearn.git
cd hsmmlearn
pip install .
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

**2. Build with CMake**

```bash
cmake -B build
cmake --build build
```

The `.so` is placed directly into `tensor_viterbi/viterbi/` — no install step needed.

- FIXING: Some errors with Anaconda GCC
- FIXING: it doesn't find pybind path sometime

**3. (Optional) Build the standalone C++ executable**

```bash
cd src
make            # produces ./tensor-viterbi
make debug      # debug build
make clean
```



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
python run_viterbi.py -m validate -dp data/3states_20steps_4dur.json

# Single timing measurement
python run_viterbi.py -m measure -dp data/20states_1000steps_20dur.json

# Benchmark (100 iterations, writes viterbi_benchmark.csv)
python run_viterbi.py -m benchmark -dp data/20states_1000steps_20dur.json
```

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
