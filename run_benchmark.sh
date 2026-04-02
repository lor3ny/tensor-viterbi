#!/bin/bash
# Usage: ./run_benchmark.sh --system <system_name> [--iterations N] [--py|--cpp|--omp|--cuda|--baseline] [--sequential]
# Available systems are defined in systems.conf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$PWD" != "$SCRIPT_DIR" ]]; then
    echo "Error: must be run from the script's directory."
    echo "  cd \"$SCRIPT_DIR\" && $0 $*"
    exit 1
fi
source "$SCRIPT_DIR/systems.conf"

# Parse arguments
SYSTEM=""
VITERBI_FLAGS=""
SEQUENTIAL=0
ITERATIONS=6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) SYSTEM="$2"; shift 2 ;;
        --py|--cpp|--omp|--cuda|--baseline)
            flag="${1#--}"
            VITERBI_FLAGS="${VITERBI_FLAGS:+$VITERBI_FLAGS:}$flag"
            shift ;;
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

if [[ -z "${SYS_TYPE[$SYSTEM]+x}" ]]; then
    echo "Error: Unknown system '$SYSTEM'."
    echo "Available systems: ${!SYS_TYPE[*]}"
    exit 1
fi

TYPE="${SYS_TYPE[$SYSTEM]}"
PARTITION="${SYS_PARTITION[$SYSTEM]}"
ACCOUNT="${SYS_ACCOUNT[$SYSTEM]}"
CPUS="${SYS_CPUS[$SYSTEM]}"
MODULES="${SYS_MODULES[$SYSTEM]}"

# Returns the wall-clock time limit for a given (states, duration, timesteps) combination.
# Rules are for the baseline; conservative enough to cover all backends.
get_walltime() {
    local s=$1 d=$2 t=$3

    if [[ $t -le 10000 ]]; then
        if [[ $s -eq 75 && $d -eq 1000 && $t -eq 10000 ]]; then
            echo "02:00:00"
        else
            echo "01:00:00"
        fi
        return
    fi

    # 100k timesteps
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
}

# Build system-specific sbatch flags (no --time, --output, --error: computed per job)
SBATCH_FLAGS=(
    "--partition=$PARTITION"
    "--account=$ACCOUNT"
    "--export=ALL,SYS_NAME=$SYSTEM,SYS_TYPE=$TYPE,SYS_MODULES=$MODULES,VITERBI_FLAGS=$VITERBI_FLAGS,BENCHMARK_ITERATIONS=$ITERATIONS"
)
if [[ "$TYPE" == "gpu" ]]; then
    SBATCH_FLAGS+=("--gres=gpu:1")
else
    SBATCH_FLAGS+=("--cpus-per-task=$CPUS")
fi

# Define parameter arrays
# states=(10 15 25 50 75)
# durations=(100 250 500 1000)
# timesteps=(1000 10000) # 100000)

states=(50)
durations=(100)
timesteps=(1000)

#states=(10)
#durations=(100 250)
#timesteps=(1000)

echo "Compiling for system: $SYSTEM"
"$SCRIPT_DIR/compile.sh" --system "$SYSTEM"
if [[ $? -ne 0 ]]; then
    echo "Error: compilation failed for $SYSTEM. Aborting."
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results/$SYSTEM"
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

            JOB_OUTPUT=$(sbatch "${SBATCH_FLAGS[@]}" \
                --job-name="tv_${stem}" \
                --time="$walltime" \
                --output="$RESULTS_DIR/${stem}.out" \
                --error="$RESULTS_DIR/${stem}.err" \
                run.slrm "$config_file")
            echo "$JOB_OUTPUT"

            if [[ "$SEQUENTIAL" -eq 1 ]]; then
                PREV_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
                echo "Waiting for job $PREV_JOB_ID to complete..."
                while squeue -j "$PREV_JOB_ID" -h &>/dev/null; do
                    sleep 30
                done
                echo "Job $PREV_JOB_ID completed."
            fi
            sleep 0.1
        done
    done
done
