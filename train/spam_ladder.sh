#!/usr/bin/env bash
# Knowledge-free ladder, rungs 3+: train each size on runs/sft/spam_v1 and score untuned vs trained on val (one-shot lists).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for spec in "mlx-community/SmolLM2-135M-Instruct smollm2-135m" "mlx-community/SmolLM2-360M-Instruct smollm2-360m" "mlx-community/Qwen3-0.6B-4bit qwen3-0.6b"; do
  set -- $spec; M=$1; T=$2
  echo "##### $T ($(date +%H:%M))"
  $PY eval/play.py --policy "mlx:$M" --system spam --tsplit val --n 8 --max-tokens 400 --batch 16 --name "spam_val_${T}_untuned" 2>&1 | grep -E '^  (val|all) '
  $PY train/sft_lora.py --model "$M" --data runs/sft/spam_v1 --out "checkpoints/spam_v1_${T}" --iters 860 --lr 1e-4 --eval-every 1000 2>&1 | grep -E 'done|rror'
  tail -1 "checkpoints/spam_v1_${T}/log.jsonl" | cut -c1-120
  $PY eval/play.py --policy "mlx:$M" --adapter "checkpoints/spam_v1_${T}" --system spam --tsplit val --n 8 --max-tokens 400 --batch 16 --name "spam_val_${T}_sft" 2>&1 | grep -E '^  (val|all) '
done
echo "finished $(date +%H:%M)"
