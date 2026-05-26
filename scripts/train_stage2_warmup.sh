#!/usr/bin/env bash
set -euo pipefail

export SKIM_STAGE="${SKIM_STAGE:-stage2_warmup}"
export ENV_FILE="${ENV_FILE:-configs/stage2_warmup.env.example}"

python code/train.py
