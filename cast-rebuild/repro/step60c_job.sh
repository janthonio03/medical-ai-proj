#!/usr/bin/env bash
# Submitted as a Slurm array: every task gets one normal RTX A5000.
set -euo pipefail
: "${STEP60C_TICKET:?Submit with repro.step60c submit/resume}"
: "${SLURM_ARRAY_JOB_ID:?Slurm array required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm worker index required}"
export PYTHONPATH="$PWD/CAST-step60c-eval${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HOME="$PWD/.cache/huggingface"
unset TRANSFORMERS_CACHE
# exec lets Slurm's TERM@180 reach Python; existing checkpoints remain immutable.
exec .envs/cast-phi/bin/python -u -m repro.step60c_worker --ticket "$STEP60C_TICKET"
