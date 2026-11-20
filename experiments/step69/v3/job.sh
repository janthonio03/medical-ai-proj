#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_ARRAY_JOB_ID:?Slurm array required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm worker required}"
export PYTHONPATH="$PWD/CAST-step60c-eval${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HOME="$PWD/.cache/huggingface"
unset TRANSFORMERS_CACHE
exec .envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py worker
