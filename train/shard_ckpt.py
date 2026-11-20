"""Store a checkpoint over 100 MB in git (GitHub's per-file limit): split adapters.safetensors into fp16 shards.

split: <dir>/adapters.safetensors -> <dir>/adapters.fp16.part<i>.safetensors (each < 90 MB), original left in place
join:  shards -> <dir>/adapters.safetensors (fp16; cast to fp32 again by sft_lora/eval if needed)
With a file name (e.g. model.safetensors for a pruned base): exact dtypes, shards in <dir>/shards/<stem>.part<i>
(a subfolder, so mlx-lm's model*.safetensors glob never sees them).
Usage: .venv/bin/python train/shard_ckpt.py split|join checkpoints/distill_t360_pipeline_pythia-70m
       .venv/bin/python train/shard_ckpt.py split|join checkpoints/spamshape4_135m_drop10 model.safetensors
"""
import sys
from pathlib import Path

import mlx.core as mx

LIMIT = 90 * 2**20


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else None
    if name:
        sd, stem = d / "shards", name.removesuffix(".safetensors")
        if mode == "split":
            w = mx.load(str(d / name)); sd.mkdir(exist_ok=True)
            parts, cur, size = [], {}, 0
            for k in sorted(w):
                if cur and size + w[k].nbytes > LIMIT:
                    parts.append(cur); cur, size = {}, 0
                cur[k] = w[k]; size += w[k].nbytes
            parts.append(cur)
            for i, p in enumerate(parts):
                mx.save_safetensors(str(sd / f"{stem}.part{i}.safetensors"), p, metadata={"format": "mlx"})
            print(f"{len(parts)} shards in {sd}")
        else:
            w = {}
            for f in sorted(sd.glob(f"{stem}.part*.safetensors")):
                w.update(mx.load(str(f)))
            mx.save_safetensors(str(d / name), w, metadata={"format": "mlx"})
            print(f"joined {len(w)} tensors -> {d / name}")
        return
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
