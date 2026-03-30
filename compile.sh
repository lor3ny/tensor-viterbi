#!/bin/bash

# Usage: ./compile.sh --system [lumi|leonardo] [--account ACCOUNT]

SYSTEM=""
ACCOUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --system|-s)
            SYSTEM="$2"
            shift 2
            ;;
        --account|-a)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --system [lumi|leonardo] [--account ACCOUNT]"
            exit 1
            ;;
    esac
done

if [[ -z "$SYSTEM" ]]; then
    echo "Error: --system argument is required."
    echo "Usage: $0 --system [lumi|leonardo] [--account ACCOUNT]"
    exit 1
fi

rm -rf build

if [[ "$SYSTEM" == "lumi" ]]; then
    [[ -z "$ACCOUNT" ]] && ACCOUNT="project_465002776"

    module load rocm
    module load cray-python

    srun --gres=gpu:1 -A "$ACCOUNT" -p standard-g cmake -B build -DGPU_PLATFORM=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx90a
    srun --gres=gpu:1 -A "$ACCOUNT" -p standard-g cmake --build build -j 8

elif [[ "$SYSTEM" == "leonardo" ]]; then
    if [[ -z "$ACCOUNT" ]]; then
        echo "Error: --account is required for Leonardo (e.g. --account IscrXX_XXXXX)"
        exit 1
    fi

    module load cuda
    module load python

    srun --gres=gpu:1 -A "$ACCOUNT" -p boost_usr_prod cmake -B build -DGPU_PLATFORM=CUDA
    srun --gres=gpu:1 -A "$ACCOUNT" -p boost_usr_prod cmake --build build -j 8

else
    echo "Error: unknown system '$SYSTEM'. Valid options: lumi, leonardo."
    exit 1
fi
