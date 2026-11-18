#!/usr/bin/env bash
# Expert-iteration RFT: each round samples from the latest adapter, adds the emulator-filtered lists to a
# growing pool (deduplicated), retrains a fresh LoRA from the base model on the pool, and evaluates.
# Usage: train/rft_loop.sh <model> <tag> <first_round> <last_round> [prev_adapter]
#   train/rft_loop.sh mlx-community/Qwen3-0.6B-4bit qwen3-0.6b 2 4 checkpoints/qwen3-0.6b_rft1
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; TAG=$2; FIRST=$3; LAST=$4; PREV=${5:-}
PY=.venv/bin/python
for r in $(seq "$FIRST" "$LAST"); do
  echo "=== round $r ($(date +%H:%M))"
  $PY train/rft.py --model "$MODEL" ${PREV:+--adapter "$PREV"} --name "rft${r}_${TAG}" --n 64 \
    | grep -E 'sampled|->'
  POOL=runs/sft/rftpool${r}_${TAG}; mkdir -p "$POOL"
  for s in train valid; do
    cat runs/sft/rft[0-9]*_"${TAG}"/$s.jsonl | $PY -c "
import sys, json
seen = set()
for l in sys.stdin:
    c = json.loads(l)['completion']
    if c not in seen:
        seen.add(c); sys.stdout.write(l)" > "$POOL/$s.jsonl"
  done
  n=$(wc -l < "$POOL/train.jsonl")
  iters=$(( n * 2 < 60 ? 60 : (n * 2 > 400 ? 400 : n * 2) ))   # ~8 epochs at batch 4, clamped
  echo "pool: $n train lists -> $iters iters"
  OUT=checkpoints/${TAG}_rft${r}
  $PY train/sft_lora.py --model "$MODEL" --data "$POOL" --out "$OUT" --iters "$iters" --lr 5e-5 \
    --eval-every 1000 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'done'
  $PY eval/play.py --policy "mlx:$MODEL" --adapter "$OUT" --n 16 --max-tokens 512 --name "rft${r}_${TAG}" \
    2>&1 | grep -E 'E=|  (train|val|test|all) '
  PREV=$OUT
done
