# Reproducing the tensor-viterbi benchmarks

This is the step-by-step guide for going from a fresh checkout to benchmark
results, on any of: a local CPU machine, a local GPU machine, a SLURM CPU
cluster, a SLURM GPU cluster, or several machines at once.

There is **no auto-detection** anywhere in this tool. `bench` only ever
learns about scheduler/system/toolchain from the YAML file you pass with
`--system`. If something isn't in that file, `bench` doesn't know it.

---

## 1. Setup

```bash
git clone https://github.com/lor3ny/tensor-viterbi.git
cd tensor-viterbi

python -m venv .venv
source .venv/bin/activate          # Linux/macOS; on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`bench` refuses to run from anywhere except the repository root, and checks
Python >= 3.10 and the packages in `requirements.txt` before doing anything
else.

---

## 2. Describe your system

Every machine (laptop, HPC login node, cloud VM) needs a YAML file under
`systems/`. Copy the template and fill in the first four fields:

```bash
cp systems/TEMPLATE.yaml systems/my-machine.yaml
```

```yaml
name: my-machine
type: cpu                # cpu | gpu
toolchain: gnu           # compiler suite: gnu | intel | amd | cray | llvm | fujitsu (cpu)
                         #                  cuda | hip (gpu)
scheduler: local         # local | slurm
```

For `scheduler: local`, delete the `slurm:` block entirely — you're done.
For `scheduler: slurm`, fill in at least `slurm.account` (and normally
`slurm.partition`):

```yaml
slurm:
  account: myproject
  partition: normal
  modules: [gcc/12.2]     # module load'ed before both build and run
```

Then validate it:

```bash
./bench check --system my-machine
```

`bench check` parses the YAML (catching typos/missing fields with a
file-and-field-specific message) and probes the environment: compiler
presence, `sbatch` if `scheduler: slurm`, `nvcc`/`hipcc` for GPU toolchains,
`likwid-perfctr` if `type: cpu`. It runs no jobs. Fix anything it flags with
`[✗]` before continuing (`[!]`/`[i]` lines are informational — e.g. a
compiler that will be provided by `module load` isn't found yet on a login
node, which is expected).

The full field reference (multi-toolchain systems, `omp_bind`/`omp_places`,
`metrics_backend`, `cc`/`cxx`, `gpu_arch`, `uenv`) is documented inline in
`systems/TEMPLATE.yaml`.

---

## 3. The universal loop

Once `bench check` is clean, every scenario below is the same three commands:

```bash
bench plan --system my-machine --pack small --cpp --omp   # build the manifest, print a preview
bench run  --system my-machine                              # execute it (compiles first)
bench status --system my-machine                             # done/running/pending/failed per job
```

`bench plan` is where you fix the backend flags (`--py`/`--cpp`/`--omp`/
`--gpu`/`--baseline`...) and `--iterations` — they're baked into each job's
manifest entry (including its walltime estimate), so `bench run` doesn't
take them at all; it just executes whatever was planned.

Re-running `bench run` is always safe: it skips jobs whose output is already
complete. Use `--force` to re-run everything, `--only-failed` to retry only
jobs whose output is missing/incomplete, `--jobs A-B` to run a 1-indexed
slice of the manifest, and `--max-hours H` (local only) to run jobs in
manifest order until the cumulative walltime estimate would exceed `H`.

---

## 4. Walkthroughs

### Local CPU

```yaml
# systems/workstation.yaml
name: workstation
type: cpu
toolchain: gnu
scheduler: local
cpus: 8
```

```bash
bench check --system workstation
bench plan --system workstation --pack small --cpp --omp --baseline
bench run  --system workstation --pack small
bench status --system workstation
```

`bench run` compiles the native extension (and the hsmmlearn baselines) with
the active Python interpreter, then calls `viterbi_app.py` directly for each
job in the manifest, one at a time.

### Local GPU

```yaml
# systems/my-gpu-box.yaml
name: my-gpu-box
type: gpu
gpu_arch: "86"          # CUDA compute capability, or a ROCm gfx code
toolchain: cuda
scheduler: local
```

```bash
bench check --system my-gpu-box
bench plan --system my-gpu-box --pack small      # defaults to --gpu on GPU systems
bench run  --system my-gpu-box --pack small
bench status --system my-gpu-box
```

### SLURM CPU

```yaml
# systems/xeon8480.yaml (shipped)
name: xeon8480
type: cpu
cpus: 112
toolchains:
  intel: { modules: [python/3.11.7, intel-oneapi-compilers] }
  gnu:   { modules: [python/3.11.7] }
slurm:
  account: IscrC_FOCAL_0
  partition: dcgp_usr_prod
```

```bash
bench check --system xeon8480 --toolchain intel
bench plan --system xeon8480 --toolchain intel --pack medium --cpp --omp --baseline
bench run  --system xeon8480 --pack medium
bench status --system xeon8480
```

`bench run` submits one `sbatch` per job (via the static `run_one.slurm`
shim) and returns immediately — it does not wait for jobs to finish. Poll
with `bench status` (which merges in `squeue --me` for running/pending
jobs).

### SLURM GPU

```yaml
# systems/a100.yaml (shipped)
name: a100
type: gpu
gpu_arch: "80"
toolchain: cuda
scheduler: slurm
slurm:
  account: IscrC_FOCAL
  partition: boost_usr_prod
  modules: [cuda, python/3.11.7]
```

```bash
bench check --system a100
bench plan --system a100 --pack small
bench run  --system a100 --pack small
bench status --system a100
```

### Multi-system reproduction

There's no shared scheduler across machines, so reproduce on each machine
independently and merge the `results/` trees afterwards — they never
collide because every system writes to its own `results/<system>/<toolchain>/`:

```bash
# On each machine:
bench plan --system <that machine's name> --pack <pack> [backend flags]
bench run  --system <that machine's name> --pack <pack>

# Then, from wherever you're aggregating results:
rsync -av user@host-a:tensor-viterbi/results/ ./results/
rsync -av user@host-b:tensor-viterbi/results/ ./results/
```

---

## 5. The pack ladder

`--pack` buckets the `states × durations × timesteps` grid
(`benchmark_params.cfg`) by estimated walltime (`walltimes.yaml`), so you can
choose how much wall-clock budget to spend without editing the grid. Old
pack names (`1h`/`2h`/`4-8h`/`10-20h`) still work as hidden aliases.

| Pack | Walltime range | Jobs (default grid) | Per-job walltime |
|---|---|---|---|
| `small`  | ≤ 1 hour   | 30 | 30 min or 1 hour |
| `medium` | 1–2 hours  | 16 | 2 hours |
| `large`  | 2–8 hours  | 8  | 4, 6, or 8 hours |
| `extra`  | 8–20 hours | 26 | 10, 14, 16, or 20 hours |

Run packs in order — `small` first to sanity-check the pipeline, then work
up. There's no single invocation that runs the whole grid.

**Local runs are strictly serial** — one job at a time, in manifest order.
`bench plan` prints a total estimated serial walltime for the pack; use:

- `--jobs A-B` to run only a slice of the manifest (e.g. split a pack across
  several terminal sessions or days: `--jobs 1-15` then `--jobs 16-30`), or
- `--max-hours H` to run as much of the pack as fits in `H` hours and report
  what's left.

SLURM runs don't have this constraint — each job is an independent `sbatch`
submission the scheduler can run concurrently, subject to your allocation.

---

## 6. LIKWID and nsys/ncu profiling

**LIKWID** (CPU-only hardware counters, fixed data file, no pack):

```bash
bench check --system workstation      # confirms likwid-perfctr is present
bench likwid --system workstation --cpp --omp
```

Output lands in the same `results/<system>/<toolchain>/` directory as
regular runs: `likwid_<version>.txt` (per-iteration LIKWID marker output)
and `likwid_<version>_<group>.csv` (one per hardware counter group). Groups
unsupported by the CPU are skipped with a `[!]` message rather than failing
the run.

**Nsight Systems / Nsight Compute** (GPU only, wraps `bench run`):

```bash
bench plan --system a100 --pack small
bench run --system a100 --pack small --nsys   # timeline trace
bench run --system a100 --pack small --ncu    # kernel-level counters (wins if both are given)
```

Profiler output is written alongside the regular `.out`/`.err`/`.csv` files
in `results/<system>/<toolchain>/`.
