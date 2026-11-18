"""LoRA DPO in MLX, using the exact eval prompt (eval/play.py render_prompt).

Data: <data>/train.jsonl rows {text, chosen, rejected}; chosen/rejected are completions in the
'<plan> ... <guesses> g1 ; g2 ; ...' format. The reference policy is the base model (no adapter); its
completion log-probs are computed once up front, before the LoRA layers are attached.

Loss per pair: -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))) + alpha * NLL(chosen)/len(chosen)
(the small SFT term keeps the chosen likelihood from collapsing, as in RPO).
Writes adapters.safetensors + adapter_config.json, loadable by mlx_lm.load(adapter_path=...).
Usage:
  .venv/bin/python train/dpo.py --model mlx-community/Qwen3-1.7B-4bit --data runs/dpo/x --out checkpoints/x
"""
import argparse, json, random, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from play import render_prompt  # noqa: E402


def encode(tok, text, completion, max_seq):
    p = tok.encode(render_prompt(tok, text), add_special_tokens=False)
    c = tok.encode(completion, add_special_tokens=False) + [tok.eos_token_id]
    return (p + c)[:max_seq], len(p)


def seq_logp(model, seq):
    """One unpadded sequence (ids, prompt_len) -> (sum log p of completion tokens, n tokens).
    Memory: only the completion positions' logits are upcast and normalised, never the whole vocab x length
    in float32 for a padded batch (that is what swapped the 16 GB Mac)."""
    ids, plen = seq
    x = mx.array(ids)[None]
    logits = model(x[:, :-1])[0, plen - 1:]          # positions predicting completion tokens
    tgt = x[0, plen:]
    lp = mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0].astype(mx.float32) \
        - mx.logsumexp(logits.astype(mx.float32), axis=-1)
    return lp.sum(), len(ids) - plen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=2, help="pairs per step")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2, help="weight of the per-token SFT term on chosen")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--max-seq", type=int, default=640)
    ap.add_argument("--mem-gb", type=float, default=5.0, help="MLX memory limit (16 GB machine: keep <= 6)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); mx.random.seed(a.seed)
    mx.set_memory_limit(int(a.mem_gb * 2**30)); mx.set_cache_limit(512 * 2**20)
    model, tok = load(a.model)
    model.freeze()
    rows = [json.loads(l) for l in open(Path(a.data) / "train.jsonl", encoding="utf-8")]
    pairs = [(encode(tok, r["text"], r["chosen"], a.max_seq), encode(tok, r["text"], r["rejected"], a.max_seq))
             for r in rows]

    # reference log-probs from the frozen base model
    t0 = time.time()
    ref = []
    for c, r in pairs:
        lc, _ = seq_logp(model, c); lr, _ = seq_logp(model, r)
        mx.eval(lc, lr)
        ref.append((lc.item(), lr.item()))
    print(f"{len(pairs)} pairs, reference log-probs in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)

    n_layers = len(model.layers)
    lora = {"rank": a.rank, "dropout": 0.0, "scale": a.scale}
    linear_to_lora_layers(model, n_layers, lora)
    print(f"trainable params: {sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6:.2f}M")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(dict(
        fine_tune_type="lora", num_layers=n_layers, lora_parameters=lora, model=a.model, data=a.data,
        method="dpo", iters=a.iters, batch_size=a.batch, learning_rate=a.lr, beta=a.beta, alpha=a.alpha,
        seed=a.seed), indent=1))

    def pair_loss(model, c, r, ref_c, ref_r):
        pc, nc = seq_logp(model, c); pr, _ = seq_logp(model, r)
        margin = a.beta * ((pc - ref_c) - (pr - ref_r))
        dpo = -nn.log_sigmoid(margin)
        return dpo + a.alpha * (-pc / nc), (dpo, margin)

    vg = nn.value_and_grad(model, lambda m, c, r, rc, rr: pair_loss(m, c, r, rc, rr)[0])
    opt = optim.Adam(learning_rate=optim.cosine_decay(a.lr, a.iters, a.lr * 0.1))
    log = open(out / "log.jsonl", "w")
    order, t0, stats = [], time.time(), []
    for it in range(1, a.iters + 1):
        if len(order) < a.batch:
            order += random.sample(range(len(pairs)), len(pairs))
        idx, order = order[:a.batch], order[a.batch:]
        grads = None
        for i in idx:  # gradient accumulation, one pair (two sequences) in memory at a time
            (loss, g) = vg(model, pairs[i][0], pairs[i][1], ref[i][0], ref[i][1])
            grads = g if grads is None else tree_map(lambda u, v: u + v, grads, g)
            mx.eval(grads, loss)
            stats.append(loss.item())
        opt.update(model, tree_map(lambda u: u / len(idx), grads))
        mx.eval(model.parameters(), opt.state)
        if it % 20 == 0 or it == a.iters:
            _, (dpo, m) = pair_loss(model, *pairs[idx[0]], *ref[idx[0]])
            info = dict(it=it, loss=sum(stats) / len(stats), dpo=dpo.item(), margin=m.item(),
                        s=round(time.time() - t0), peak_gb=round(mx.get_peak_memory() / 2**30, 2))
            log.write(json.dumps(info) + "\n"); log.flush(); stats = []
            print(info, flush=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
