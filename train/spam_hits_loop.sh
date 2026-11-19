#!/usr/bin/env bash
# Guess-level RFT rounds (train/spam_hits.py). Usage: spam_hits_loop.sh MODEL TAG START_ADAPTER LAST_ROUND [FIRST_ROUND]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M=$1; T=$2; PREV=$3; R=$4; S=${5:-1}
for r in $(seq "$S" "$R"); do
  echo "=== $T hits round $r ($(date +%H:%M))"
  $PY train/spam_hits.py --model "$M" --adapter "$PREV" --teacher runs/sft/spam_v2 --name "spamhits${r}_$T" 2>&1 | grep -E 'hit rate|rror'
  n=$(wc -l < "runs/sft/spamhits${r}_$T/train.jsonl"); iters=$(( n / 2 ))
  OUT=checkpoints/spamhits${r}_$T
  $PY train/sft_lora.py --model "$M" --data "runs/sft/spamhits${r}_$T" --out "$OUT" --iters "$iters" --lr 1e-4 --eval-every 100000 2>&1 | grep -E 'done|rror'
  $PY eval/play.py --policy "mlx:$M" --adapter "$OUT" --system spam --tsplit val --n 8 --max-tokens 480 --batch 16 --name "spamhits${r}_val_$T" 2>&1 | grep -E '^  val '
  PREV=$OUT
done
