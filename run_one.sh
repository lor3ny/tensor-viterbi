#!/bin/bash
# run_one.sh — runs exactly one benchmark job. Used identically by both
# backends:
#   - local:  bench invokes this directly as a subprocess
#   - slurm:  run_one.slurm (the sbatch shim) execs this
#
# Everything is passed in via environment variables so both call sites stay
# byte-identical:
#   SYS_NAME SYS_TYPE SYS_MODULES SYS_UENV SYS_OMP_BIND SYS_OMP_PLACES
#   SYS_CPUS SYS_METRICS_BACKEND VITERBI_FLAGS BENCHMARK_ITERATIONS DATA_PATH
#   NSYS_PROFILE NCU_PROFILE RESULTS_DIR JOB_STEM
#
# Under sbatch, SLURM_SUBMIT_DIR/SLURM_CPUS_PER_TASK/SLURM_JOB_NAME are also
# present and take priority where noted below (same as the old
# generate_slrm() heredoc). Deliberately no `set -e`/`set -u`: the original
# heredoc ran without strict mode (e.g. a non-matching lscpu grep must not
# abort the job before the actual benchmark runs).

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -n "${SYS_UENV:-}" && -z "${_UENV_ACTIVE:-}" ]] && command -v uenv &>/dev/null; then
    exec uenv run --view=modules "$SYS_UENV" -- env _UENV_ACTIVE=1 bash "$0" "$@"
fi

if [[ "$SYS_TYPE" == "cpu" ]]; then
    lscpu | grep -E "Model name|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket"
    grep MemTotal /proc/meminfo
elif [[ "$SYS_TYPE" == "gpu" ]]; then
    command -v nvidia-smi &>/dev/null && \
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    command -v rocm-smi &>/dev/null && \
        rocm-smi --showproductname --showmeminfo vram --showdriverversion
fi

if command -v module &>/dev/null; then
    IFS=':' read -ra _MODS <<< "${SYS_MODULES:-}"
    for _mod in "${_MODS[@]}"; do [[ -n "$_mod" ]] && module load "$_mod"; done
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
elif [[ -n "${SYS_CPUS:-}" ]]; then
    export OMP_NUM_THREADS=$SYS_CPUS
fi
[[ -n "${SYS_OMP_BIND:-}" ]]   && export OMP_PROC_BIND="$SYS_OMP_BIND"
[[ -n "${SYS_OMP_PLACES:-}" ]] && export OMP_PLACES="$SYS_OMP_PLACES"

cd "$SCRIPT_DIR"

IFS='/' read -r _SYS _TC <<< "$SYS_NAME"
PYTHON_FLAGS=""
IFS=':' read -ra _VFLAGS <<< "${VITERBI_FLAGS:-}"
for _f in "${_VFLAGS[@]}"; do [[ -n "$_f" ]] && PYTHON_FLAGS="$PYTHON_FLAGS --$_f"; done

PROFILE_LABEL="${SLURM_JOB_NAME:-${JOB_STEM:-benchmark}}"

RUNNER=""
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
    RUNNER="nsys profile -f true -o ${RESULTS_DIR}/${PROFILE_LABEL}"
fi
if [[ "${NCU_PROFILE:-0}" == "1" ]]; then
    RUNNER="ncu --target-processes all --set full --force-overwrite --launch-skip 1500 --launch-count 10 -o ${RESULTS_DIR}/${PROFILE_LABEL}"
fi

$RUNNER python viterbi_app.py $PYTHON_FLAGS \
    --system "$_SYS" --toolchain "$_TC" \
    --iterations "${BENCHMARK_ITERATIONS:-6}" \
    --data-path "$DATA_PATH"
