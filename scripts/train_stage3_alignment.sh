#!/usr/bin/env bash
set -euo pipefail

export SKIM_STAGE="${SKIM_STAGE:-stage3_alignment}"
export ENV_FILE="${ENV_FILE:-configs/stage3_alignment.env.example}"

python code/train.py
