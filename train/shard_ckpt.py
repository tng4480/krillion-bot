"""Store a checkpoint over 100 MB in git (GitHub's per-file limit): split adapters.safetensors into fp16 shards.

split: <dir>/adapters.safetensors -> <dir>/adapters.fp16.part<i>.safetensors (each < 90 MB), original left in place
join:  shards -> <dir>/adapters.safetensors (fp16; cast to fp32 again by sft_lora/eval if needed)
Usage: .venv/bin/python train/shard_ckpt.py split|join checkpoints/distill_t360_pipeline_pythia-70m
"""
import sys
from pathlib import Path

import mlx.core as mx

LIMIT = 90 * 2**20


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    if mode == "split":
        w = mx.load(str(d / "adapters.safetensors"))
        parts, cur, size = [], {}, 0
        for k in sorted(w):
            v = w[k].astype(mx.float16) if w[k].dtype == mx.float32 else w[k]
            if cur and size + v.nbytes > LIMIT:
                parts.append(cur); cur, size = {}, 0
            cur[k] = v; size += v.nbytes
        parts.append(cur)
        for i, p in enumerate(parts):
            mx.save_safetensors(str(d / f"adapters.fp16.part{i}.safetensors"), p)
        print(f"{len(parts)} shards in {d}")
    else:
        w = {}
        for f in sorted(d.glob("adapters.fp16.part*.safetensors")):
            w.update(mx.load(str(f)))
        mx.save_safetensors(str(d / "adapters.safetensors"), w)
        print(f"joined {len(w)} tensors -> {d / 'adapters.safetensors'}")


if __name__ == "__main__":
    main()
