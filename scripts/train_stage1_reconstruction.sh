#!/usr/bin/env bash
set -euo pipefail

export SKIM_STAGE="${SKIM_STAGE:-stage1_reconstruction}"
export ENV_FILE="${ENV_FILE:-configs/stage1_reconstruction.env.example}"

python code/train.py
