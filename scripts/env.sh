#!/usr/bin/env bash

export UV_PROJECT_ENVIRONMENT="/goinfre/$USER/deepracer-racing-lab/.venv"
export UV_CACHE_DIR="/goinfre/$USER/deepracer-racing-lab/uv-cache"
export TMPDIR="/goinfre/$USER/deepracer-racing-lab/tmp"
export MPLCONFIGDIR="/goinfre/$USER/deepracer-racing-lab/matplotlib"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$TMPDIR" \
    "$MPLCONFIGDIR"