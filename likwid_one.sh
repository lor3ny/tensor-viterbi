#!/bin/bash
# likwid_one.sh — the LIKWID hardware-counter profiling flow, used
# identically by `bench likwid` locally and via sbatch (likwid_one.slurm).
# Replaces the old duplicated generate_likwid_slrm()/run_likwid_local() pair.
#
# Env vars: SYS_NAME SYS_MODULES SYS_UENV SYS_OMP_BIND SYS_OMP_PLACES
#           SYS_CPUS RESULTS_DIR DATA_PATH LIKWID_GROUPS LIKWID_VERSION_FLAGS
# No `set -e`/`set -u` — same reasoning as run_one.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -n "${SYS_UENV:-}" && -z "${_UENV_ACTIVE:-}" ]] && command -v uenv &>/dev/null; then
    exec uenv run --view=modules "$SYS_UENV" -- env _UENV_ACTIVE=1 bash "$0" "$@"
fi

lscpu | grep -E "Model name|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket"
grep MemTotal /proc/meminfo

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
OUTDIR="${RESULTS_DIR}"
mkdir -p "$OUTDIR"

declare -a PERF_GROUPS=()
for _g in ${LIKWID_GROUPS:-}; do
    if likwid-perfctr -a 2>/dev/null | grep -qE "^[[:space:]]*${_g}[[:space:]]"; then
        PERF_GROUPS+=("$_g")
    else
        echo "[!] Skipping group $_g: not available on this CPU" >&2
    fi
done
if [[ ${#PERF_GROUPS[@]} -eq 0 ]]; then
    echo "[!] No requested LIKWID groups are available on this CPU." >&2
fi

for VERSION_FLAG in ${LIKWID_VERSION_FLAGS:-}; do
    VERSION_NAME="${VERSION_FLAG#--}"
    OUTPUT_FILE="${OUTDIR}/likwid_${VERSION_NAME}.txt"
    > "$OUTPUT_FILE"

    for GROUP in "${PERF_GROUPS[@]}"; do
        echo "-> ${VERSION_NAME} / ${GROUP}"
        LIKWID_CORES="E:N:${OMP_NUM_THREADS}"
        LIKWID_CSV="${OUTDIR}/likwid_${VERSION_NAME}_${GROUP}.csv"

        likwid-perfctr -C "${LIKWID_CORES}" -g "${GROUP}" -m \
            -o "${LIKWID_CSV}" \
            -- python "${SCRIPT_DIR}/viterbi_app.py" \
               --system "${_SYS}_likwid" --toolchain "${_TC}" \
               --iterations 1 \
               "${VERSION_FLAG}" \
               --data-path "${DATA_PATH}" \
            >> "$OUTPUT_FILE" \
            2>>"${OUTDIR}/likwid.log"
    done
done
