#!/usr/bin/env bash
# The full spam training program for one model, as used for SmolLM2 135M/360M:
#   SFT on runs/sft/spam_v2 -> guess-level RFT rounds 1-3 (train/spam_hits.py) -> round 4 with --shape -> val + test.
# Each round retrains from the base model. Usage:
#   train/size_pipeline.sh MODEL TAG "SFT_FLAGS" [FIRST_STEP]
#   e.g. train/size_pipeline.sh EleutherAI/pythia-70m pythia-70m "--full --lr 3e-5 --warmup 60 --mem-gb 3"
# FIRST_STEP: sft | 1 | 2 | 3 | 4 | test (resume after a crash).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M=$1; T=$2; F=$3; FIRST=${4:-sft}
step() { echo "=== $T $1 ($(date +%H:%M))"; }
need() { [ -s "$1" ] || { echo "!!! missing $1, stopping"; exit 1; }; }
ev() { $PY eval/play.py --policy "mlx:$M" --adapter "$1" --system spam --tsplit "$2" --n 8 --max-tokens 480 --batch 16 \
        --name "$3" 2>&1 | grep -E "^  (val|test) "; }
order="sft 1 2 3 4 test"; go=0
for s in $order; do
  [ "$s" = "$FIRST" ] && go=1
  [ $go = 1 ] || continue
  case $s in
    sft)
      step sft
      $PY train/sft_lora.py --model "$M" --data runs/sft/spam_v2 --out "checkpoints/spam_v2_$T" --iters 860 \
          --eval-every 100000 $F 2>&1 | grep -E 'done|rror'
      need "checkpoints/spam_v2_$T/adapters.safetensors"
      ev "checkpoints/spam_v2_$T" val "spam_v2_val_$T" ;;
    1|2|3|4)
      prev=$([ "$s" = 1 ] && echo "checkpoints/spam_v2_$T" || echo "checkpoints/spamhits$((s - 1))_$T")
      base=$([ "$s" = 4 ] && echo "spamshape4" || echo "spamhits${s}"); name="${base}_$T"
      shape=$([ "$s" = 4 ] && echo "--shape" || echo "")
      step "round $s ($name)"
      $PY train/spam_hits.py --model "$M" --adapter "$prev" --teacher runs/sft/spam_v2 --name "$name" $shape 2>&1 \
          | grep -E 'hit rate|rror'
      need "runs/sft/$name/train.jsonl"
      n=$(wc -l < "runs/sft/$name/train.jsonl")
      $PY train/sft_lora.py --model "$M" --data "runs/sft/$name" --out "checkpoints/$name" --iters $(( n / 2 )) \
          --eval-every 100000 $F 2>&1 | grep -E 'done|rror'
      need "checkpoints/$name/adapters.safetensors"
      ev "checkpoints/$name" val "${base}_val_$T" ;;
    test)
      step test
      ev "checkpoints/spamshape4_$T" test "test_spamshape4_$T" ;;
  esac
done
echo "=== $T finished ($(date +%H:%M))"
