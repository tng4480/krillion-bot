#!/usr/bin/env bash
# Distillation students: train fresh from base on runs/sft/distill_<NAME>_{raw,pipeline}, score val + test.
# Usage: train/distill_queue.sh NAME LANE   (LANE = smol | pythia)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; N=$1; LANE=$2
ev() { $PY eval/play.py --policy "mlx:$1" --adapter "$2" --system spam --tsplit "$3" --n 8 --max-tokens 480 \
        --batch 16 --name "$4" 2>&1 | grep -E "^  (val|test) "; }
run() {  # MODEL TAG KIND FLAGS
  local M=$1 T=$2 K=$3 F=$4 D=runs/sft/distill_${N}_$3 O=checkpoints/distill_${N}_$3_$2
  local n=$(wc -l < "$D/train.jsonl")
  echo "=== $T $K: train $(( n / 2 )) iters ($(date +%H:%M))"
  $PY train/sft_lora.py --model "$M" --data "$D" --out "$O" --iters $(( n / 2 )) --eval-every 100000 $F 2>&1 \
      | grep -E 'done|rror'
  [ -s "$O/adapters.safetensors" ] || { echo "!!! $O failed"; return; }
  echo "  val:"; ev "$M" "$O" val "distill_${N}_${K}_val_$T"
  echo "  test:"; ev "$M" "$O" test "test_distill_${N}_${K}_$T"
}
if [ "$LANE" = smol ]; then
  for K in pipeline raw; do run mlx-community/SmolLM2-135M-Instruct smollm2-135m $K "--lr 1e-4 --mem-gb 3"; done
else
  for K in pipeline raw; do run EleutherAI/pythia-14m pythia-14m $K "--full --lr 1e-4 --warmup 60 --mem-gb 3"; done
  run EleutherAI/pythia-70m pythia-70m pipeline "--full --lr 3e-5 --warmup 60 --mem-gb 3"
fi
echo "=== lane $LANE finished ($(date +%H:%M))"
