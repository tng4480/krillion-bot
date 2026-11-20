"""Prune a trained Pythia's vocabulary to the tokens the spam task uses: a smaller model, same transformer.

91.5% of Pythia-14M is its two 50,304 x 128 token tables; the transformer is 1.2M params. This keeps
  - the tokens seen in the distillation data (train prompts rendered as at play time + completions), or only the
    --top K most frequent of them,
  - every token on their BPE merge chains (so the cut tokenizer still reaches them),
  - the 256 single-byte tokens (so any text still encodes) and <|endoftext|> / <|padding|>,
renumbers them in old-id order, slices embed_in / embed_out to those rows, and writes a plain mlx-lm model folder
(config, model.safetensors, a cut tokenizer.json). With every seen token kept, training text tokenizes exactly as
before (checked); with --top K, rarer words split into smaller pieces, so heal with a short retrain on the same data.
Usage: .venv/bin/python train/prune_vocab.py --weights checkpoints/distill_t360_pipeline_pythia-14m --out checkpoints/vocab_all_pythia-14m [--top 8000]
"""
import argparse, collections, json, shutil, sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
from play import render_prompt  # noqa: E402

DATA = ["runs/sft/distill_t360_pipeline", "runs/sft/distill_t360_raw"]  # built from train prompts only


def texts(tok):
    for s in DATA:
        for f in ("train", "valid"):
            for l in open(ROOT / s / f"{f}.jsonl"):
                d = json.loads(l)
                yield render_prompt(tok, d["text"], d.get("system", "spam")), d["completion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-14m")
    ap.add_argument("--weights", required=True, help="full fine-tune checkpoint dir (adapters.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=0, help="keep only the K most frequent seen tokens (0 = all seen)")
    a = ap.parse_args()
    src, out = Path(snapshot_download(a.base)), ROOT / a.out
    tok = AutoTokenizer.from_pretrained(src)
    cnt = collections.Counter()
    for p, c in texts(tok):
        cnt.update(tok.encode(p, add_special_tokens=False)); cnt.update(tok.encode(c, add_special_tokens=False))
    seen = [i for i, _ in (cnt.most_common(a.top) if a.top else cnt.most_common())]

    tj = json.load(open(src / "tokenizer.json"))
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    id2s = {i: s for s, i in vocab.items()}
    parent = {}  # token -> the merge (a, b) that makes it
    for m in merges:
        x, y = m.split(" ") if isinstance(m, str) else m
        parent.setdefault(x + y, (x, y))
    keep = {i for s, i in vocab.items() if len(s) == 1}  # the 256 byte-level base chars
    keep |= {t["id"] for t in tj["added_tokens"] if t["special"]}
    stack = [id2s[i] for i in seen if i in id2s] + [t["content"] for t in tj["added_tokens"] if t["id"] in set(seen)]
    keep |= {i for i in seen}
    while stack:  # merge-chain closure
        s = stack.pop()
        if s in parent:
            for q in parent[s]:
                if vocab[q] not in keep:
                    keep.add(vocab[q]); stack.append(q)
    old = sorted(keep)
    new = {o: n for n, o in enumerate(old)}

    tj["model"]["vocab"] = {s: new[i] for s, i in vocab.items() if i in new}
    nv = tj["model"]["vocab"]
    pairs = [m.split(" ") if isinstance(m, str) else m for m in merges]
    tj["model"]["merges"] = [m for m, (x, y) in zip(merges, pairs) if x in nv and y in nv and x + y in nv]
    tj["added_tokens"] = [dict(t, id=new[t["id"]]) for t in tj["added_tokens"] if t["id"] in new]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(tj, open(out / "tokenizer.json", "w"), ensure_ascii=False)
    tc = json.load(open(src / "tokenizer_config.json"))
    tc["added_tokens_decoder"] = {str(new[int(k)]): v for k, v in tc["added_tokens_decoder"].items() if int(k) in new}
    json.dump(tc, open(out / "tokenizer_config.json", "w"), indent=1)
    shutil.copy(src / "special_tokens_map.json", out)
    cfg = json.load(open(src / "config.json"))
    cfg.update(vocab_size=len(old), bos_token_id=new[0], eos_token_id=new[0])
    json.dump(cfg, open(out / "config.json", "w"), indent=1)

    w = mx.load(str(ROOT / a.weights / "adapters.safetensors"))
    idx = mx.array(old)
    for k in ("model.embed_in.weight", "model.embed_out.weight"):
        w[k] = w[k][idx]
    mx.save_safetensors(str(out / "model.safetensors"), w, metadata={"format": "mlx"})
    n_par = sum(v.size for v in w.values())
    json.dump(dict(weights=a.weights, top=a.top, vocab=len(old), seen=len(seen), params=n_par),
              open(out / "prune.json", "w"), indent=1)
    print(f"vocab {len(old)} (seen kept {len(seen)}, + bytes/specials/merge chains), params {n_par / 1e6:.2f}M")

    t2 = AutoTokenizer.from_pretrained(out)  # check: training text tokenizes the same, renumbered
    same = tot = 0
    for p, c in list(texts(tok))[::50]:
        for s in (p, c):
            a1 = [new.get(i, -1) for i in tok.encode(s, add_special_tokens=False)]
            same += a1 == t2.encode(s, add_special_tokens=False); tot += 1
    print(f"identical tokenization on {same}/{tot} sampled training texts; eos {t2.eos_token_id}")


if __name__ == "__main__":
    main()
