#!/usr/bin/env bash
# Score depth-pruned versions of the trained 135M (train/prune.py), then heal each with a short LoRA fine-tune on the
# same data that trained it (runs/sft/spamshape4_smollm2-135m) and score again. Val for all; test at the end.
# Usage: train/prune_pipeline.sh TAG "6 10 15" [DATA]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; T=$1; DROPS=$2; DATA=${3:-runs/sft/spamshape4_smollm2-135m}
ev() { $PY eval/play.py --policy "mlx:$1" ${2:+--adapter "$2"} --system spam --tsplit "$3" --n 8 --max-tokens 480 \
        --batch 16 --name "$4" 2>&1 | grep -E "^  (val|test) "; }
n=$(wc -l < "$DATA/train.jsonl")
for d in $DROPS; do
  M=checkpoints/${T}_drop$d
  echo "=== drop $d: no heal ($(date +%H:%M))"; ev "$M" "" val "prune_${T}_drop${d}_val"
  echo "=== drop $d: heal ($(date +%H:%M))"
  $PY train/sft_lora.py --model "$M" --data "$DATA" --out "checkpoints/heal_${T}_drop$d" --iters $(( n / 2 )) \
      --lr 1e-4 --eval-every 100000 --mem-gb 3 2>&1 | grep -E 'done|rror'
  [ -s "checkpoints/heal_${T}_drop$d/adapters.safetensors" ] || { echo "!!! heal failed"; exit 1; }
  ev "$M" "checkpoints/heal_${T}_drop$d" val "prune_${T}_drop${d}_heal_val"
done
for d in $DROPS; do
  echo "=== drop $d: test, healed ($(date +%H:%M))"
  ev "checkpoints/${T}_drop$d" "checkpoints/heal_${T}_drop$d" test "test_prune_${T}_drop${d}_heal"
done
echo "=== finished ($(date +%H:%M))"
