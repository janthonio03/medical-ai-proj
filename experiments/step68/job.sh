#!/usr/bin/env bash
set -euo pipefail
: "${STEP60C_TICKET:?Missing submission ticket}"
: "${SLURM_ARRAY_JOB_ID:?Slurm array required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm worker index required}"
export PYTHONPATH="$PWD/CAST-step60c-eval${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HOME="$PWD/.cache/huggingface"
unset TRANSFORMERS_CACHE
exec .envs/cast-phi/bin/python -B -u "$PWD/CAST-step68-neutral-resume/resume.py" worker "$STEP60C_TICKET"
