#!/bin/bash
# Usage: ./run_benchmark.sh --system <system_name> [--iterations N] [--py|--cpp|--omp|--omp-opt|--cuda|--baseline] [--sequential]
# Available systems are defined in systems.conf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$PWD" != "$SCRIPT_DIR" ]]; then
    echo "Error: must be run from the script's directory."
    echo "  cd \"$SCRIPT_DIR\" && $0 $*"
    exit 1
fi
source "$SCRIPT_DIR/systems.conf"
_ORIG_ARGS=("$@")

# Parse arguments
SYSTEM=""
TOOLCHAIN=""
VITERBI_FLAGS=""
SEQUENTIAL=0
LOCAL=0
ITERATIONS=6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)    SYSTEM="$2";    shift 2 ;;
        --toolchain) TOOLCHAIN="$2"; shift 2 ;;
        --py|--cpp|--omp|--omp-opt|--cuda|--baseline|--baseline-cpp|--baseline-omp)
            flag="${1#--}"
            VITERBI_FLAGS="${VITERBI_FLAGS:+$VITERBI_FLAGS:}$flag"
            shift ;;
        --local)      LOCAL=1;      shift ;;
        --sequential) SEQUENTIAL=1; shift ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SYSTEM" ]]; then
    echo "Error: --system argument is required."
    echo "Available systems: ${!SYS_TYPE[*]}"
    exit 1
fi

if [[ -z "$TOOLCHAIN" ]]; then
    echo "Error: --toolchain argument is required."
    exit 1
fi

if [[ -z "${SYS_TYPE[$SYSTEM]+x}" ]]; then
    echo "Error: Unknown system '$SYSTEM'."
    echo "Available systems: ${!SYS_TYPE[*]}"
    exit 1
fi

# --toolchain all: re-exec for every toolchain defined for this system
if [[ "$TOOLCHAIN" == "all" ]]; then
    _toolchains=$(for k in "${!SYS_MODULES[@]}"; do [[ "$k" == "$SYSTEM/"* ]] && echo "${k#*/}"; done | sort)
    if [[ -z "$_toolchains" ]]; then
        echo "Error: No toolchains defined for system '$SYSTEM'."
        exit 1
    fi
    # Rebuild original args without --toolchain
    _orig_args=()
    _skip=0
    for _a in "$@"; do
        if [[ $_skip -eq 1 ]]; then _skip=0; continue; fi
        if [[ "$_a" == "--toolchain" ]]; then _skip=1; continue; fi
        _orig_args+=("$_a")
    done
    for _tc in $_toolchains; do
        echo "=== Submitting $SYSTEM / $_tc ==="
        "$0" "${_orig_args[@]}" --toolchain "$_tc"
    done
    exit $?
fi

if [[ -z "${SYS_MODULES[$SYSTEM/$TOOLCHAIN]+x}" ]]; then
    echo "Error: Toolchain '$TOOLCHAIN' is not defined for system '$SYSTEM'."
    _known=$(for k in "${!SYS_MODULES[@]}"; do [[ "$k" == "$SYSTEM/"* ]] && echo "  ${k#*/}"; done | sort)
    echo "Known toolchains for $SYSTEM:${_known:- (none)}"
    exit 1
fi

TYPE="${SYS_TYPE[$SYSTEM]}"
PARTITION="${SYS_PARTITION[$SYSTEM]}"
QOS="${SYS_QOS[$SYSTEM]:-}"
ACCOUNT="${SYS_ACCOUNT[$SYSTEM]}"
CPUS="${SYS_CPUS[$SYSTEM]}"
OMP_BIND="${SYS_OMP_BIND[$SYSTEM]:-}"
OMP_PLACES="${SYS_OMP_PLACES[$SYSTEM]:-}"

# Default for CPU: run all variants if no flags were specified
if [[ "$TYPE" == "cpu" && -z "$VITERBI_FLAGS" ]]; then
    VITERBI_FLAGS="cpp:omp-opt:baseline-cpp:baseline-omp"
fi

MODULES="${SYS_MODULES[$SYSTEM/$TOOLCHAIN]}"
METRICS_BACKEND="${SYS_METRICS_BACKEND[$SYSTEM/$TOOLCHAIN]:-}"
UENV="${SYS_UENV[$SYSTEM/$TOOLCHAIN]:-}"
SYS_NAME="$SYSTEM/$TOOLCHAIN"

# If a uenv is required and we are not already inside it, re-exec under it.
# This ensures the venv symlinks and module commands resolve correctly.
if [[ -n "$UENV" && -z "${_UENV_ACTIVE:-}" ]]; then
    exec uenv run --view=modules "$UENV" -- \
        env _UENV_ACTIVE=1 bash "$0" "${_ORIG_ARGS[@]}"
fi

# Returns the wall-clock time limit for a given (states, duration, timesteps) combination.
# Rules are for the baseline; conservative enough to cover all backends.
get_walltime() {
    local s=$1 d=$2 t=$3

    if [[ $t -le 10000 ]]; then
        if [[ $s -eq 75 && $d -eq 1000 && $t -eq 10000 ]]; then
            echo "02:00:00"
        else
            echo "00:30:00"
        fi
        return
    fi

    if [[ $t -eq 100000 ]]; then
        if [[ $s -le 15 ]]; then
            echo "01:00:00"
        elif [[ $s -eq 25 ]]; then
            if   [[ $d -le 250 ]]; then echo "01:00:00"
            else                        echo "02:00:00"
            fi
        elif [[ $s -eq 50 ]]; then
            if   [[ $d -eq 100  ]]; then echo "01:00:00"
            elif [[ $d -eq 250  ]]; then echo "02:00:00"
            elif [[ $d -eq 500  ]]; then echo "04:00:00"
            else                         echo "08:00:00"
            fi
        elif [[ $s -eq 75 ]]; then
            if   [[ $d -eq 100  ]]; then echo "02:00:00"
            elif [[ $d -eq 250  ]]; then echo "04:00:00"
            elif [[ $d -eq 500  ]]; then echo "08:00:00"
            else                         echo "16:00:00"
            fi
        else
            echo "01:00:00"
        fi
        return
    fi

    if [[ $t -eq 1000000 ]]; then
        if [[ $s -le 15 ]]; then
            echo "02:00:00"
        elif [[ $s -eq 25 ]]; then
            if   [[ $d -le 250 ]]; then echo "02:00:00"
            else                        echo "06:00:00"
            fi
        elif [[ $s -eq 50 ]]; then
            if   [[ $d -eq 100  ]]; then echo "2:00:00"
            elif [[ $d -eq 250  ]]; then echo "8:00:00"
            elif [[ $d -eq 500  ]]; then echo "14:00:00"
            else                         echo "20:00:00"
            fi
        elif [[ $s -eq 75 ]]; then
            if   [[ $d -eq 100  ]]; then echo "5:00:00"
            elif [[ $d -eq 250  ]]; then echo "10:00:00"
            elif [[ $d -eq 500  ]]; then echo "20:00:00"
            else                         echo "20:00:00"
            fi
        else
            echo "02:00:00"
        fi
        return
    fi

    if [[ $t -eq 10000000 ]]; then
        echo "20:00:00"
        return
    fi
}

# Build system-specific sbatch flags (no --time, --output, --error, --export: computed per job)
SBATCH_FLAGS=(
    "--account=$ACCOUNT"
)
[[ -n "$PARTITION" ]] && SBATCH_FLAGS+=("--partition=$PARTITION")
[[ -n "$QOS" ]] && SBATCH_FLAGS+=("--qos=$QOS")
if [[ "$TYPE" == "gpu" ]]; then
    SBATCH_FLAGS+=("--gres=gpu:1")
else
    SBATCH_FLAGS+=("--cpus-per-task=$CPUS")
fi

# submit_job <stem> <viterbi_flags> <iterations> <walltime> <config_file>
#   stem         : used for job name, .out/.err paths (may include a suffix like _baseline_cpp)
#   viterbi_flags: colon-separated list (e.g. "baseline-cpp" or "omp:cpp")
#   iterations   : number of benchmark iterations
submit_job() {
    local job_stem="$1" vflags="$2" iters="$3" walltime="$4" config_file="$5"
    local JOB_OUTPUT
    echo "  -> flags=[$vflags] iterations=$iters"
    if [[ "$LOCAL" -eq 1 ]]; then
        SYS_NAME="$SYS_NAME" SYS_TYPE="$TYPE" SYS_MODULES="$MODULES" \
        SYS_METRICS_BACKEND="$METRICS_BACKEND" SYS_UENV="$UENV" SYS_CPUS="$CPUS" \
        SYS_OMP_BIND="$OMP_BIND" SYS_OMP_PLACES="$OMP_PLACES" \
        VITERBI_FLAGS="$vflags" BENCHMARK_ITERATIONS="$iters" \
        bash "$SCRIPT_DIR/run.slrm" "$config_file" \
            > "$RESULTS_DIR/${job_stem}.out" 2> "$RESULTS_DIR/${job_stem}.err"
        return
    fi
    local export_str="ALL,SYS_NAME=$SYS_NAME,SYS_TYPE=$TYPE,SYS_MODULES=$MODULES,SYS_METRICS_BACKEND=$METRICS_BACKEND,SYS_UENV=$UENV,SYS_OMP_BIND=$OMP_BIND,SYS_OMP_PLACES=$OMP_PLACES,VITERBI_FLAGS=$vflags,BENCHMARK_ITERATIONS=$iters"
    JOB_OUTPUT=$(sbatch "${SBATCH_FLAGS[@]}" \
        "--export=$export_str" \
        --job-name="tv_${job_stem}" \
        --time="$walltime" \
        --output="$RESULTS_DIR/${job_stem}.out" \
        --error="$RESULTS_DIR/${job_stem}.err" \
        run.slrm "$config_file")
    echo "$JOB_OUTPUT"
    if [[ "$SEQUENTIAL" -eq 1 ]]; then
        PREV_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
        echo "Waiting for job $PREV_JOB_ID to complete..."
        while squeue -j "$PREV_JOB_ID" -h &>/dev/null; do sleep 30; done
        echo "Job $PREV_JOB_ID completed."
    fi
    sleep 0.1
}

# Define parameter arrays
#states=(10 15 25 50 75)
#durations=(100 250 500 1000)
#timesteps=(1000000)

states=(10 15 25 50 75)
durations=(100 250 500 1000)
timesteps=(100000) # 100000)

#states=(100)
#durations=(10000)
#timesteps=(10000000)


# Pre-flight: verify that compile.sh has already been run for this system/toolchain.
VENV_DIR="$SCRIPT_DIR/.venv/$SYS_NAME"
SO_FILE="$SCRIPT_DIR/tensor_viterbi/viterbi/$SYS_NAME/_native.so"
if [[ ! -f "$VENV_DIR/bin/python3" ]]; then
    echo "Error: no virtual environment found at $VENV_DIR."
    echo "  Run: ./compile.sh --system $SYSTEM --toolchain $TOOLCHAIN"
    exit 1
fi
if [[ ! -f "$SO_FILE" ]]; then
    echo "Error: native extension not found at $SO_FILE."
    echo "  Run: ./compile.sh --system $SYSTEM --toolchain $TOOLCHAIN"
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results/$SYS_NAME"
mkdir -p "$RESULTS_DIR"

# Loop through all combinations
PREV_JOB_ID=""
for s in "${states[@]}"; do
    for d in "${durations[@]}"; do
        for t in "${timesteps[@]}"; do
            walltime="$(get_walltime "$s" "$d" "$t")"
            stem="${s}s_${d}d_${t}t"
            config_file="data/${s}states_${t}steps_${d}dur.json"
            echo "Submitting job for: System=$SYSTEM, State=$s, Duration=$d, Timesteps=$t, Walltime=$walltime"

            # Reduce iterations for very large T to keep wall-time reasonable.
            _iters=$ITERATIONS
            [[ $t -eq 1000000  && $_iters -gt 2 ]] && _iters=2
            [[ $t -eq 10000000 && $_iters -gt 2 ]] && _iters=2

            if [[ "$TYPE" == "cpu" && $t -eq 100000 ]]; then
                # At T=100000, HSMMLearn C++ needs fewer iterations (very slow).
                # Split into two jobs: baseline-cpp (2 iter) + everything else (_iters).
                # Determine whether baseline-cpp is part of the current run.
                _runs_bcpp=0
                _rest_flags=()
                if [[ -z "$VITERBI_FLAGS" ]]; then
                    # No flags = run all: split baseline-cpp out; rest runs baseline-omp + cpp + omp
                    _runs_bcpp=1
                    _rest_flags=("baseline-omp" "cpp" "omp")
                else
                    IFS=':' read -ra _vf <<< "$VITERBI_FLAGS"
                    for _f in "${_vf[@]}"; do
                        if [[ "$_f" == "baseline" ]]; then
                            _runs_bcpp=1
                            _rest_flags+=("baseline-omp")   # baseline = cpp+omp; keep omp in rest
                        elif [[ "$_f" == "baseline-cpp" ]]; then
                            _runs_bcpp=1
                        else
                            _rest_flags+=("$_f")
                        fi
                    done
                fi
                if [[ $_runs_bcpp -eq 1 ]]; then
                    submit_job "${stem}_baseline_cpp" "baseline-cpp" "2" "$walltime" "$config_file"
                fi
                if [[ ${#_rest_flags[@]} -gt 0 ]]; then
                    _rest=$(IFS=':'; echo "${_rest_flags[*]}")
                    submit_job "$stem" "$_rest" "$_iters" "$walltime" "$config_file"
                fi
            else
                submit_job "$stem" "$VITERBI_FLAGS" "$_iters" "$walltime" "$config_file"
            fi
        done
    done
done
