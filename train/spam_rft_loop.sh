#!/usr/bin/env bash
# RFT rounds for the spam students: sample from the current adapter on train prompts, keep good + diverse lists,
# retrain a fresh LoRA from base on kept + teacher lists, evaluate on val. Usage: spam_rft_loop.sh MODEL TAG START_ADAPTER ROUNDS
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M=$1; T=$2; PREV=$3; R=$4
for r in $(seq 1 "$R"); do
  echo "=== $T round $r ($(date +%H:%M))"
  $PY train/spam_rft.py --model "$M" --adapter "$PREV" --teacher runs/sft/spam_v2 --name "spamrft${r}_$T" 2>&1 | grep -E 'kept|rror'
  n=$(wc -l < "runs/sft/spamrft${r}_$T/train.jsonl"); iters=$(( n / 2 ))   # ~2 epochs at batch 4
  OUT=checkpoints/spamrft${r}_$T
  $PY train/sft_lora.py --model "$M" --data "runs/sft/spamrft${r}_$T" --out "$OUT" --iters "$iters" --lr 1e-4 --eval-every 100000 2>&1 | grep -E 'done|rror'
  $PY eval/play.py --policy "mlx:$M" --adapter "$OUT" --system spam --tsplit val --n 8 --max-tokens 480 --batch 16 --name "spamrft${r}_val_$T" 2>&1 | grep -E '^  val '
  PREV=$OUT
done
