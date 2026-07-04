"""Train small_llm on shakespeare / openwebtext, nanoGPT-style.

nanoGPT trains each dataset separately (no joint config); we match:

  shakespeare (BPE, from scratch — analogue of nanoGPT's train_shakespeare_char):
    cd data/shakespeare && uv run python prepare.py && cd -
    uv run python train_small_llm.py --config configs/shakespeare.json \\
        --data_dir data/shakespeare --out_dir out/shakespeare \\
        --batch_size 64 --grad_accum_steps 1 --max_iters 5000 \\
        --warmup_iters 100 --lr_decay_iters 5000 --min_lr 1e-4 \\
        --eval_interval 250 --eval_iters 200

  openwebtext (GPT-2 124M reproduction — nanoGPT's config/train_gpt2.py):
    cd data/openwebtext && uv run python prepare.py && cd -
    uv run python train_small_llm.py --config configs/openwebtext.json \\
        --data_dir data/openwebtext --out_dir out/gpt2-owt
        # defaults already match: batch=12, grad_accum=40 (eff. batch 480),
        # max_iters=600000, warmup=2000, min_lr=6e-5, eval_interval=1000.

Reuses `small_llm.llm.LLM`'s model + loss but drives its own optimizer
(warmup + cosine LR, AdamW β₂=0.95, grad clip, grad accumulation) and its own
train loop. Logs go to stdout and to `<out_dir>/metrics.jsonl` (one JSON record
per line). No wandb / trackio — the notebook `eval.ipynb` plots from the jsonl.
"""
import argparse
import dataclasses
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from small_llm.llm import LLM, ModelConfig
from small_llm.data import DataLoader, ensure_bin


def build_optimizer(peak_lr, min_lr, warmup_iters, decay_iters, weight_decay, grad_clip):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=warmup_iters,
        decay_steps=decay_iters, end_value=min_lr,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=weight_decay),
    )
    return optimizer, schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out")
    # nanoGPT train_gpt2.py defaults (openwebtext / GPT-2 124M):
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--grad_accum_steps", type=int, default=40,
                   help="effective batch = batch_size * grad_accum_steps (nanoGPT: 480)")
    p.add_argument("--max_iters", type=int, default=600_000)
    p.add_argument("--warmup_iters", type=int, default=2000)
    p.add_argument("--lr_decay_iters", type=int, default=None,
                   help="defaults to max_iters (nanoGPT convention)")
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=10)
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    train_bin = ensure_bin(args.data_dir, 'train')
    val_bin = ensure_bin(args.data_dir, 'val')
    train_loader = DataLoader(train_bin, args.batch_size, cfg.max_len, seed=cfg.seed)
    val_loader = DataLoader(val_bin, args.batch_size, cfg.max_len, seed=cfg.seed + 1)

    llm = LLM(cfg)
    decay_iters = args.lr_decay_iters or args.max_iters
    optimizer, schedule = build_optimizer(
        cfg.learning_rate, args.min_lr, args.warmup_iters, decay_iters,
        cfg.weight_decay, args.grad_clip,
    )
    state = llm.model_state.replace(opt_state=optimizer.init(llm.model_state.params), step=0)
    loss_and_grad = llm.loss_fn
    model = llm.model

    @jax.jit
    def micro_grads(params, batch, rng):
        x, y = batch[:, :-1], batch[:, 1:]
        return loss_and_grad(params, rng, x, y)

    @jax.jit
    def apply_grads(state, grads):
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(params=params, opt_state=opt_state, step=state.step + 1)

    @jax.jit
    def eval_step(params, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model.apply(params, x, training=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    def estimate_val_loss():
        losses = [float(eval_step(state.params, next(val_loader))) for _ in range(args.eval_iters)]
        return float(np.mean(losses))

    log_path = os.path.join(args.out_dir, 'metrics.jsonl')
    log_file = open(log_path, 'a')
    print(f"config: {dataclasses.asdict(cfg)}", flush=True)
    print(f"effective batch = {args.batch_size} * {args.grad_accum_steps} = "
          f"{args.batch_size * args.grad_accum_steps}", flush=True)
    print("compiling first step (this can take ~30-60s on CPU)...", flush=True)

    t0 = time.time()
    best_val = float('inf')
    for step in range(args.max_iters):
        grads_sum = None
        loss_sum = 0.0
        for _ in range(args.grad_accum_steps):
            batch = jnp.asarray(next(train_loader))
            loss, grads = micro_grads(state.params, batch, llm._next_rng())
            grads_sum = grads if grads_sum is None else jax.tree.map(jnp.add, grads_sum, grads)
            loss_sum += float(loss)
        grads_mean = jax.tree.map(lambda g: g / args.grad_accum_steps, grads_sum)
        state = apply_grads(state, grads_mean)
        train_loss = loss_sum / args.grad_accum_steps

        if step % args.log_interval == 0:
            lr = float(schedule(step))
            rec = {"step": step, "train_loss": train_loss, "lr": lr, "time": time.time() - t0}
            log_file.write(json.dumps(rec) + "\n"); log_file.flush()
            print(f"step {step}: train_loss={train_loss:.4f} lr={lr:.2e} "
                  f"elapsed={rec['time']:.0f}s", flush=True)

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            val_loss = estimate_val_loss()
            log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n"); log_file.flush()
            print(f"step {step}: val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {"params": state.params, "step": int(state.step),
                        "config": dataclasses.asdict(cfg), "val_loss": val_loss}
                with open(os.path.join(args.out_dir, "ckpt.pkl"), "wb") as f:
                    pickle.dump(ckpt, f)

    log_file.close()


if __name__ == "__main__":
    main()
