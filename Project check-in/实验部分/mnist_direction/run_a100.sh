#!/usr/bin/env bash
# One-seed closed loop on a Jetstream2 GPU instance. Run inside tmux from this directory:
#   bash run_a100.sh [seed]
# Steps: environment check -> official-loop validation -> short c=1 gate -> full run (ref, c=1, c=2, c=0.5) -> figures.
# Each step stops the script if it fails. Outputs go to generated/direction/seed<seed>/.
if [ -f /workspace/env.sh ]; then source /workspace/env.sh; fi
set -euo pipefail
cd "$(dirname "$0")"
SEED="${1:-0}"
OUT="generated/direction/seed${SEED}"
DATA="data"
mkdir -p "$OUT"

python -c "import torchvision" 2>/dev/null || uv pip install torchvision
python -c "import matplotlib" 2>/dev/null || uv pip install matplotlib
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "== 0. training primitives vs the official notebook loop (50 steps) =="
python ../mnist_dynamics/code/run_dynamics.py --experiment notebook --root "$OUT/official_validation" --data-root "$DATA" --validate

echo "== 1. gate: c=1 must replay the reference (2,000 steps) =="
rm -rf "$OUT/gate"
python code/run_direction.py --root "$OUT/gate" --data-root "$DATA" --seed "$SEED" --cs 1 --steps 2000 2>&1 | tee "$OUT/gate.log"

echo "== 2. full run: reference + c=1, 2, 0.5, 100,000 updates after the fork =="
python code/run_direction.py --root "$OUT/full" --data-root "$DATA" --seed "$SEED" --cs 1 2 0.5 --steps 100000 2>&1 | tee "$OUT/full.log"

echo "== 3. figures and first-pass analysis =="
python code/make_direction_figures.py --root "$OUT/full"
echo "DONE: $OUT/full  (pull the results, then shelve the instance: js2 stop)"
