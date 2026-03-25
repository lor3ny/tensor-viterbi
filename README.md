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
├── CMakeLists.txt
└── pyproject.toml
```

## How to Start

### Requirements

- Python >= 3.10
- numpy
- deprecated
  
#### (optional) if you want also native versions:
- CUDA toolkit *>=* 12.0
- pybind11 *>=* 2.12 
- scikit-build-core *>=* 0.9



### Installation

#### Python only (no C++/CUDA)

```bash
pip install numpy deprecated
```

Then import directly, the native extension is optional and gracefully skipped if not built:

```python
from tensor_viterbi import HSMM, decode_log_tensor_viterbi_cached
```

#### With C++/CUDA extension

**1. Install build dependencies**

```bash
pip install scikit-build-core pybind11
```

**2. Build and install in editable mode**

```bash
pip install -e .
```

This runs CMake, compiles `src/bindings.cpp` + the CUDA kernels, and places `_native.so` inside `tensor_viterbi/viterbi/`.

**3. (Optional) Build the standalone C++ executable**

```bash
cd src
make            # produces ./tensor-viterbi
make debug      # debug build
make clean
```

> The Makefile targets NVIDIA A100 (`sm_80`). Edit `ARCH` in `src/Makefile` or `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs.



## How to Run

### As a library

```python
from tensor_viterbi import HSMM, decode_log_tensor_viterbi_cached, decode_vanilla_viterbi

hsmm = HSMM.load_model("data/20states_1000steps_20dur.json")

path = decode_log_tensor_viterbi_cached(hsmm)
```

### CLI runner

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
