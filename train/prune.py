"""Depth-prune a (fused) llama-style model: drop the block of n consecutive layers that changes the hidden state least.

1. Fuse the LoRA adapter into the base weights (mlx_lm fuse) -> checkpoints/fused_<tag>.
2. Calibration: run --calib train completions (prompt + completion, the format the model plays in) and record the
   hidden state entering every layer. For each block [l, l+n), the angular distance between the state entering
   layer l and the state entering layer l+n, averaged over tokens (Gromov et al. 2024, "The Unreasonable
   Ineffectiveness of the Deeper Layers"). The block with the smallest distance is the one the model needs least.
3. Write checkpoints/<tag>_drop<n>: same weights without those layers, renumbered, config updated.
The pruned dirs are ordinary MLX models: eval with --policy mlx:<dir>, heal with train/sft_lora.py --model <dir>.
Usage: .venv/bin/python train/prune.py --model mlx-community/SmolLM2-135M-Instruct \
           --adapter checkpoints/spamshape4_smollm2-135m --tag spamshape4_135m --drop 6,10,15
"""
import argparse, json, random, re, shutil, subprocess, sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from play import render_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to fuse first (omit to prune the base model)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--drop", default="6,10,15")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--data", default="runs/sft/spamshape4_smollm2-135m/train.jsonl")
    a = ap.parse_args()
    mx.set_memory_limit(3 * 2**30)
    src = ROOT / "checkpoints" / f"fused_{a.tag}"
    if not (src / "config.json").exists():
        if a.adapter:
            subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", a.model, "--adapter-path", a.adapter,
                            "--save-path", str(src)], check=True)
        else:  # prune the base model itself: copy its MLX snapshot
            from huggingface_hub import snapshot_download
            shutil.copytree(snapshot_download(a.model), src, dirs_exist_ok=True)
    from mlx_lm import load
    model, tok = load(str(src))
    layers = model.model.layers
    L = len(layers)
    rows = [json.loads(l) for l in open(ROOT / a.data, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    dist_sum = {}
    n_tok = 0
    drops = [int(x) for x in a.drop.split(",")]
    for d in rows[:a.calib]:
        ids = tok.encode(render_prompt(tok, d["text"], d.get("system", "spam")) + d["completion"],
                         add_special_tokens=False)[:768]
        h = model.model.embed_tokens(mx.array([ids]))
        states = [h]
        for layer in layers:
            h = layer(h, "causal", None)
            states.append(h)
        S = [s[0].astype(mx.float32) for s in states]
        S = [s / mx.linalg.norm(s, axis=-1, keepdims=True) for s in S]
        for n in drops:
            for l in range(1, L - n + 1):   # never drop layer 0 (embedding interface)
                cos = mx.clip(mx.sum(S[l] * S[l + n], axis=-1), -1, 1)
                ang = mx.arccos(cos).sum() / 3.14159265
                dist_sum[(n, l)] = dist_sum.get((n, l), 0.0) + ang.item()
        n_tok += len(ids)
    weights = mx.load(str(src / "model.safetensors")) if (src / "model.safetensors").exists() else {}
    if not weights:
        for f in sorted(src.glob("model*.safetensors")):
            weights.update(mx.load(str(f)))
    cfg = json.loads((src / "config.json").read_text())
    for n in drops:
        best = min(range(1, L - n + 1), key=lambda l: dist_sum[(n, l)])
        gone = set(range(best, best + n))
        curve = " ".join(f"{l}:{dist_sum[(n, l)] / n_tok:.3f}" for l in range(1, L - n + 1, max(1, (L - n) // 8)))
        print(f"drop {n}: best block {best}-{best + n - 1} (angular dist {dist_sum[(n, best)] / n_tok:.3f}); "
              f"by start {curve}", flush=True)
        keep = [i for i in range(L) if i not in gone]
        remap = {old: new for new, old in enumerate(keep)}
        out = {}
        for k, v in weights.items():
            m = re.match(r"(model\.layers\.)(\d+)(\..*)", k)
            if m:
                i = int(m.group(2))
                if i in gone:
                    continue
                k = f"{m.group(1)}{remap[i]}{m.group(3)}"
            out[k] = v
        dst = ROOT / "checkpoints" / f"{a.tag}_drop{n}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
                shutil.copy(f, dst / f.name)
        c = dict(cfg, num_hidden_layers=len(keep))
        if "layer_types" in c and isinstance(c["layer_types"], list):
            c["layer_types"] = [c["layer_types"][i] for i in keep]
        (dst / "config.json").write_text(json.dumps(c, indent=1))
        mx.save_safetensors(str(dst / "model.safetensors"), out, metadata={"format": "mlx"})
        n_par = sum(v.size for v in out.values())
        print(f"  -> {dst} ({len(keep)} layers, {n_par / 1e6:.1f}M params)", flush=True)


if __name__ == "__main__":
    main()
