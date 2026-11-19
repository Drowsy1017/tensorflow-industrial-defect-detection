#!/usr/bin/env bash

source .venv/bin/activate

export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "TensorFlow environment ready."
