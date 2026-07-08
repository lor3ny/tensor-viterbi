#!/bin/bash
# likwid_one.slurm — minimal sbatch shim for LIKWID profiling, committed as a
# static file. Job-specific flags and --export are injected by `bench
# likwid` on the sbatch command line; the work happens in likwid_one.sh,
# shared byte-for-byte with the local backend.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
exec "${SLURM_SUBMIT_DIR:-.}/likwid_one.sh"
