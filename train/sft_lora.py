"""LoRA SFT with mlx-lm's trainer, using the exact eval prompt (eval/play.py render_prompt) and prompt masking.

mlx-lm's built-in chat datasets re-apply the chat template without enable_thinking=False and without our
system prompt handling, so training and eval prompts would differ. Here each example is
render_prompt(text) + completion + EOS, and the loss covers only the completion.

The adapter dir is loadable by mlx_lm.load(model, adapter_path=...) and eval/play.py --adapter.
Usage:
  .venv/bin/python train/sft_lora.py --model mlx-community/Qwen3-0.6B-4bit --data runs/sft/v1 \
      --out checkpoints/qwen3-0.6b_sft-v1 --iters 400
"""
import argparse, json, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from play import render_prompt  # noqa: E402


class SFTDataset:
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok

    def process(self, d):
        p = self.tok.encode(render_prompt(self.tok, d["text"], d.get("system", "plan")), add_special_tokens=False)
        c = self.tok.encode(d["completion"], add_special_tokens=False) + [self.tok.eos_token_id]
        return (p + c, len(p))

    def __getitem__(self, i):
        return self.rows[i]

    def __len__(self):
        return len(self.rows)


class Log(TrainingCallback):
    def __init__(self, path):
        self.f = open(path, "a")

    def on_train_loss_report(self, info):
        self.f.write(json.dumps({"kind": "train", **info}) + "\n"); self.f.flush()

    def on_val_loss_report(self, info):
        self.f.write(json.dumps({"kind": "val", **info}) + "\n"); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--layers", type=int, default=-1, help="-1 = all layers")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--resume", default=None, help="adapters.safetensors to continue from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=5.0, help="MLX memory limit (16 GB machine: keep <= 6)")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--warmup", type=int, default=0, help="linear warmup steps before the cosine decay")
    ap.add_argument("--full", action="store_true", help="full fine-tune (all weights) instead of LoRA; for tiny models")
    a = ap.parse_args()

    mx.random.seed(a.seed)
    mx.set_memory_limit(int(a.mem_gb * 2**30)); mx.set_cache_limit(512 * 2**20)
    model, tok = load(a.model)
    n_layers = len(model.layers) if a.layers < 0 else a.layers
    lora = {"rank": a.rank, "dropout": 0.0, "scale": a.scale}
    if a.full:
        model.set_dtype(mx.float32)   # fp16/bf16 full fine-tunes go NaN (Pythia ships fp16)
        model.unfreeze()
    else:
        model.freeze()
        linear_to_lora_layers(model, n_layers, lora)
    if a.resume:
        model.load_weights(a.resume, strict=False)
    n_train = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.2f}M")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(dict(
        fine_tune_type="full" if a.full else "lora", num_layers=n_layers, lora_parameters=lora, model=a.model,
        data=a.data, iters=a.iters, batch_size=a.batch, learning_rate=a.lr, seed=a.seed), indent=1))

    train_set = SFTDataset(Path(a.data) / "train.jsonl", tok)
    valid_set = SFTDataset(Path(a.data) / "valid.jsonl", tok)
    targs = TrainingArgs(batch_size=a.batch, iters=a.iters, val_batches=-1, steps_per_report=20,
                         steps_per_eval=a.eval_every, steps_per_save=a.eval_every,
                         adapter_file=str(out / "adapters.safetensors"), max_seq_length=a.max_seq,
                         grad_checkpoint=not a.no_grad_checkpoint)
    sched = optim.cosine_decay(a.lr, a.iters - a.warmup, a.lr * 0.1)
    if a.warmup:
        sched = optim.join_schedules([optim.linear_schedule(a.lr * 0.01, a.lr, a.warmup), sched], [a.warmup])
    opt = optim.Adam(learning_rate=sched)
    t0 = time.time()
    train(model=model, args=targs, optimizer=opt, train_dataset=CacheDataset(train_set),
          val_dataset=CacheDataset(valid_set), training_callback=Log(out / "log.jsonl"))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
