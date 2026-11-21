"""Record, for every MLP neuron of the model, its strongest firings over the corpus.

Models: dense gated MLPs, e.g. Gemma 3 (GeGLU), Qwen3.5, Mistral 3 / Ministral 3 (SwiGLU).
Neuron activation: h = act(g) * u, with g = gate_proj(x), u = up_proj(x) (h is the input of down_proj).
sign(act(g)) == sign(g) for GELU and SiLU, so every firing falls in one sign regime. Per neuron we keep 3 lists:
  top_pos_pos   g > 0, u > 0      the K largest h
  top_neg_neg   g < 0, u < 0      the K largest h  ("double negative")
  bottom_mixed  opposite signs    the K most negative h
Only tokens with mask.bin == 1 are ranked (not the first token of a window, chat-template tokens or padding).
A firing is stored as its index in tokens.bin; merge_and_export.py turns it into (doc_id, pos).

Parallelism (srun): rank r processes windows r, r + world, ... on --gpus_per_rank GPUs. With more than one GPU
(24-27B models) the decoder layers are split over them, and the buffers of each layer stay on that layer's GPU.
Each rank checkpoints to <out_dir>/rank_<r>.pt; rerunning resumes from there (keep the same number of ranks).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

LISTS = ["top_pos_pos", "top_neg_neg", "bottom_mixed"]


class TopK:
    """For one layer: per list and neuron, the K highest scores seen so far, sorted (-inf = empty slot).
    The score of bottom_mixed is -h, so its highest scores are the most negative activations."""

    def __init__(self, d_ff, k, dtype, id_dtype, device):
        self.vals = torch.full((len(LISTS), d_ff, k), float("-inf"), dtype=dtype, device=device)
        self.ids = torch.full((len(LISTS), d_ff, k), -1, dtype=id_dtype, device=device)

    def update(self, h, g, u, token_ids):
        """h, g, u: [n_tokens, d_ff]; token_ids: [n_tokens], index of each token in tokens.bin."""
        sign = g.sign() * u.sign()  # 0 if g or u is exactly 0: such tokens go in no list
        scores = [
            torch.where((sign > 0) & (g > 0), h, float("-inf")),  # top_pos_pos
            torch.where((sign > 0) & (g < 0), h, float("-inf")),  # top_neg_neg
            torch.where(sign < 0, -h, float("-inf")),             # bottom_mixed
        ]
        for i, score in enumerate(scores):
            # only the tokens beating a neuron's current K-th best can enter its list
            n = int((score > self.vals[i, :, -1]).sum(0).max())
            if n == 0:
                continue
            new_vals, rows = score.topk(n, dim=0)  # [n, d_ff]
            vals = torch.cat([self.vals[i], new_vals.T], dim=1)
            ids = torch.cat([self.ids[i], token_ids[rows].T.to(self.ids.dtype)], dim=1)
            self.vals[i], order = vals.topk(self.vals.shape[-1], dim=1)
            self.ids[i] = ids.gather(1, order)


def load_decoder(model_path, dtype, gpus):
    """The text decoder of the model on the given GPUs (its layers split evenly over them if more than one)."""
    is_multimodal = hasattr(AutoConfig.from_pretrained(model_path), "vision_config")
    model_class = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    if len(gpus) == 1:
        model = model_class.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").to(gpus[0])
    else:
        # cap every GPU at ~1/n of the weights, so that device_map="auto" spreads the layers evenly
        weights = sum(os.path.getsize(f"{model_path}/{f}") for f in os.listdir(model_path) if f.endswith(".safetensors"))
        max_memory = {gpu: int(1.15 * weights / len(gpus)) for gpu in gpus}
        model = model_class.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa",
                                            device_map="auto", max_memory=max_memory)
    return model.get_decoder().eval()


def make_batches(lengths, tokens_per_batch):
    """Group window indices, longest first, so that n_windows * longest window <= tokens_per_batch."""
    batches = []
    for i in np.argsort(-lengths, kind="stable"):
        longest = lengths[batches[-1][0]] if batches else 0
        if batches and (len(batches[-1]) + 1) * longest <= tokens_per_batch:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="where the per-rank checkpoints go")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--tokens_per_batch", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ckpt_minutes", type=float, default=60.0)
    ap.add_argument("--gpus_per_rank", type=int, default=1, help="GPUs per srun task; the model is split over them")
    args = ap.parse_args()

    # this rank and its GPUs
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    if torch.cuda.is_available():
        assert args.gpus_per_rank <= torch.cuda.device_count(), "--gpus_per_rank is larger than the visible GPUs"
        first = local_rank * args.gpus_per_rank % torch.cuda.device_count()
        gpus = list(range(first, first + args.gpus_per_rank))
    else:
        gpus = ["cpu"]

    # this rank's windows of the corpus
    meta = json.load(open(f"{args.corpus_dir}/meta.json"))
    windows = pd.read_parquet(f"{args.corpus_dir}/windows.parquet", columns=["offset", "length"]).iloc[rank::world]
    offsets, lengths = windows.offset.to_numpy(), windows.length.to_numpy()
    tokens = np.memmap(f"{args.corpus_dir}/tokens.bin", dtype=np.uint32, mode="r")
    mask = np.memmap(f"{args.corpus_dir}/mask.bin", dtype=np.uint8, mode="r")
    batches = make_batches(lengths, args.tokens_per_batch)

    # model, and one TopK per layer on the GPU of that layer
    decoder = load_decoder(args.model_path, getattr(torch, args.dtype), gpus)
    layers = decoder.layers
    d_ff = layers[0].mlp.down_proj.in_features
    id_dtype = torch.int32 if meta["n_tokens"] < 2**31 else torch.int64
    topk = [TopK(d_ff, args.k, getattr(torch, args.dtype), id_dtype, layer.mlp.down_proj.weight.device) for layer in layers]

    # the MLPs feed the buffers; `current` holds the rankable-token mask and token ids of the batch, per device
    current = {}

    def recording_forward(mlp, layer_topk):
        def forward(x):
            g, u = mlp.gate_proj(x), mlp.up_proj(x)
            h = mlp.act_fn(g) * u
            rankable, token_ids = current[h.device]
            layer_topk.update(h[rankable], g[rankable], u[rankable], token_ids)
            return mlp.down_proj(h)
        return forward

    for layer, layer_topk in zip(layers, topk):
        layer.mlp.forward = recording_forward(layer.mlp, layer_topk)

    # resume from this rank's checkpoint, if any
    ckpt = f"{args.out_dir}/rank_{rank}.pt"
    state = {"batches_done": 0, "tokens_done": 0, "done": False}
    if os.path.exists(ckpt):
        state = torch.load(ckpt)
        for layer_topk, vals, ids in zip(topk, state.pop("vals"), state.pop("ids")):
            layer_topk.vals.copy_(vals)
            layer_topk.ids.copy_(ids)
        if state["done"]:
            print(f"[rank {rank}] already done")
            return

    def save(done):
        """Write the checkpoint (vals, ids: [L, 3, d_ff, K] on cpu) to a .tmp file, then rename it: a crash never
        leaves a broken checkpoint."""
        os.makedirs(args.out_dir, exist_ok=True)
        state.update(done=done, world=world, k=args.k, model_path=args.model_path)
        vals = torch.stack([t.vals.cpu() for t in topk])
        ids = torch.stack([t.ids.cpu() for t in topk])
        torch.save({**state, "vals": vals, "ids": ids}, f"{ckpt}.tmp")
        os.replace(f"{ckpt}.tmp", ckpt)

    start = last_save = time.time()
    tokens_at_start = state["tokens_done"]
    input_device = decoder.get_input_embeddings().weight.device
    with torch.inference_mode():
        for b in range(state["batches_done"], len(batches)):
            # right-padded batch of windows
            batch = batches[b]
            shape = (len(batch), lengths[batch[0]])
            input_ids, attention, rankable, token_ids = (np.zeros(shape, t) for t in (np.int64, np.int64, bool, np.int64))
            for row, w in enumerate(batch):
                o, n = offsets[w], lengths[w]
                input_ids[row, :n] = tokens[o:o + n]
                attention[row, :n] = 1
                rankable[row, :n] = mask[o:o + n]
                token_ids[row, :n] = np.arange(o, o + n)
            current.clear()
            for device in {t.vals.device for t in topk}:
                current[device] = (torch.from_numpy(rankable).to(device), torch.from_numpy(token_ids[rankable]).to(device))

            decoder(input_ids=torch.from_numpy(input_ids).to(input_device),
                    attention_mask=torch.from_numpy(attention).to(input_device), use_cache=False)
            state["batches_done"] = b + 1
            state["tokens_done"] += int(rankable.sum())

            if b % 20 == 0:
                rate = (state["tokens_done"] - tokens_at_start) / (time.time() - start)
                print(f"[rank {rank}] batch {b + 1}/{len(batches)}, {state['tokens_done']:,} tokens ({rate:,.0f} tok/s)", flush=True)
            if time.time() - last_save > args.ckpt_minutes * 60:
                save(done=False)
                last_save = time.time()
    save(done=True)
    print(f"[rank {rank}] done: {state['tokens_done']:,} tokens in {time.time() - start:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
