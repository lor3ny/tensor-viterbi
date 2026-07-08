#!/bin/bash
# run_one.slurm — minimal sbatch shim, committed as a static file (bench no
# longer writes a .tmp_benchmark.slrm at runtime). All job-specific flags
# (--job-name, --output, --error, --partition, --account, --qos, --time,
# --cpus-per-task/--gres) and the environment (--export) are injected by
# `bench run` on the sbatch command line. The actual work happens in
# run_one.sh, shared byte-for-byte with the local backend.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
exec "${SLURM_SUBMIT_DIR:-.}/run_one.sh"
