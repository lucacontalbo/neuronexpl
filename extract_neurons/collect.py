"""Record, for every MLP neuron of a gated-MLP model, its strongest firings over the corpus.
Works with gemma-3-4b-it (GeGLU) and Qwen3.5 (SwiGLU).

Activation: h = act(g) * u, with g = gate_proj(x), u = up_proj(x) (the input of down_proj).
Since sign(act(g)) == sign(g) for GELU and SiLU, every firing has a sign regime. Per neuron we keep:
  top_pos_pos   g > 0, u > 0      the K largest h
  top_neg_neg   g < 0, u < 0      the K largest h  ("double negative")
  bottom_mixed  opposite signs    the K most negative h
Only tokens with mask.bin == 1 count (no first token of a window, chat-template tokens or padding).
A firing is stored as its index in tokens.bin; merge_and_export.py turns it into (doc_id, pos).

One process per GPU (srun): rank r handles windows r, r + world, ... and checkpoints
its buffers to <out_dir>/rank_<r>.pt; rerunning resumes from there.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText


class TopK:
    """Per layer, list and neuron: the K highest scores seen so far, sorted, -inf = empty.
    The bottom_mixed score is -h, so its highest scores are the most negative h."""

    def __init__(self, n_layers, d_ff, k, dtype, id_dtype, device):
        self.vals = torch.full((n_layers, 3, d_ff, k), float("-inf"), dtype=dtype, device=device)
        self.ids = torch.full((n_layers, 3, d_ff, k), -1, dtype=id_dtype, device=device)

    def update(self, layer, h, g, u, token_ids):
        """h, g, u: [n_tokens, d_ff]; token_ids: [n_tokens] index of each token in tokens.bin."""
        sign = g.sign() * u.sign()
        scores = [
            torch.where((sign > 0) & (g > 0), h, float("-inf")),
            torch.where((sign > 0) & (g < 0), h, float("-inf")),
            torch.where(sign < 0, -h, float("-inf")),
        ]
        for i, s in enumerate(scores):
            vals, ids = self.vals[layer, i], self.ids[layer, i]
            # Only tokens beating a neuron's current K-th best can enter its list.
            n = int((s > vals[:, -1]).sum(0).max())
            if n == 0:
                continue
            new_vals, rows = s.topk(n, dim=0)  # [n, d_ff]
            merged, order = torch.cat([vals, new_vals.T], 1).topk(vals.shape[1], dim=1)
            self.ids[layer, i] = torch.cat([ids, token_ids[rows].T.to(ids.dtype)], 1).gather(1, order)
            self.vals[layer, i] = merged


def make_batches(lengths, tokens_per_batch):
    """Group window indices, longest first, so that n_windows * longest <= tokens_per_batch."""
    batches = []
    for i in np.argsort(-lengths, kind="stable"):
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= tokens_per_batch:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--tokens_per_batch", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ckpt_minutes", type=float, default=60.0)
    args = ap.parse_args()
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    device = f"cuda:{int(os.environ.get('SLURM_LOCALID', 0)) % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    # This rank's windows of the corpus.
    meta = json.load(open(f"{args.corpus_dir}/meta.json"))
    win = pd.read_parquet(f"{args.corpus_dir}/windows.parquet", columns=["offset", "length"]).iloc[rank::world]
    offsets, lengths = win.offset.to_numpy(), win.length.to_numpy()
    tokens = np.memmap(f"{args.corpus_dir}/tokens.bin", dtype=np.uint32, mode="r")
    mask = np.memmap(f"{args.corpus_dir}/mask.bin", dtype=np.uint8, mode="r")
    batches = make_batches(lengths, args.tokens_per_batch)

    # Text decoder (e.g. Gemma3TextModel, Qwen3_5TextModel), with MLPs patched to feed the top-K buffers.
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=dtype, attn_implementation="sdpa")
    lm = model.get_decoder().to(device).eval()
    id_dtype = torch.int32 if meta["n_tokens"] < 2**31 else torch.int64
    topk = TopK(lm.config.num_hidden_layers, lm.config.intermediate_size, args.k, dtype, id_dtype, device)
    batch = {}  # rankable-token mask and token ids of the current batch

    def patched_forward(mlp, layer):
        def forward(x):
            g, u = mlp.gate_proj(x), mlp.up_proj(x)
            h = mlp.act_fn(g) * u
            m = batch["mask"]
            topk.update(layer, h[m], g[m], u[m], batch["token_ids"])
            return mlp.down_proj(h)
        return forward

    for layer, block in enumerate(lm.layers):
        block.mlp.forward = patched_forward(block.mlp, layer)

    # Resume from this rank's checkpoint, if any.
    ckpt = f"{args.out_dir}/rank_{rank}.pt"
    state = {"batches_done": 0, "tokens_done": 0, "done": False}
    if os.path.exists(ckpt):
        state = torch.load(ckpt)
        topk.vals.copy_(state.pop("vals"))
        topk.ids.copy_(state.pop("ids"))
        if state["done"]:
            print(f"[rank {rank}] already done")
            return

    def save(done):  # write to .tmp then rename, so a crash never leaves a broken checkpoint
        os.makedirs(args.out_dir, exist_ok=True)
        state.update(done=done, world=world, k=args.k, model_path=args.model_path)
        torch.save({**state, "vals": topk.vals.cpu(), "ids": topk.ids.cpu()}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    t0 = last_save = time.time()
    tokens_at_start = state["tokens_done"]
    with torch.inference_mode():
        for b in range(state["batches_done"], len(batches)):
            idx = batches[b]
            shape = (len(idx), lengths[idx[0]])
            input_ids, attn, keep, token_ids = (np.zeros(shape, dt) for dt in (np.int64, np.int64, bool, np.int64))
            for j, i in enumerate(idx):
                o, n = offsets[i], lengths[i]
                input_ids[j, :n], attn[j, :n], keep[j, :n], token_ids[j, :n] = tokens[o:o + n], 1, mask[o:o + n], np.arange(o, o + n)
            batch["mask"] = torch.from_numpy(keep).to(device)
            batch["token_ids"] = torch.from_numpy(token_ids[keep]).to(device)
            lm(input_ids=torch.from_numpy(input_ids).to(device), attention_mask=torch.from_numpy(attn).to(device), use_cache=False)
            state["batches_done"], state["tokens_done"] = b + 1, state["tokens_done"] + int(keep.sum())

            if b % 20 == 0:
                rate = (state["tokens_done"] - tokens_at_start) / (time.time() - t0)
                print(f"[rank {rank}] batch {b + 1}/{len(batches)}, {state['tokens_done']:,} tokens ({rate:,.0f} tok/s)", flush=True)
            if time.time() - last_save > args.ckpt_minutes * 60:
                save(done=False)
                last_save = time.time()
    save(done=True)
    print(f"[rank {rank}] done: {state['tokens_done']:,} tokens in {time.time() - t0:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
