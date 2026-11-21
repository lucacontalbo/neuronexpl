"""Merge the per-rank checkpoints of collect.py into the neuron map: one file per layer and list,
  <out_dir>/layer_XX/{top_pos_pos,top_neg_neg,bottom_mixed}.npy
each an array [d_ff, K] of records (doc_id, pos, activation): row n = neuron n, column = rank, strongest first
(bottom_mixed: most negative first). Read it with show.load_list, or np.load(path, mmap_mode="r")[n].
pos is the token index in the tokenized document (no BOS), so same doc_id = same input.
A list has fewer than K records if its regime fired fewer than K times in the corpus: the empty slots, at the end
of the row, have doc_id = -1, pos = -1, activation = nan.
"""
import argparse
import glob
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch

from collect import LISTS

SIGN = {"top_pos_pos": 1, "top_neg_neg": 1, "bottom_mixed": -1}  # activation = sign * score of collect.py
RECORD = np.dtype([("doc_id", "<i4"), ("pos", "<i4"), ("activation", "<f4")])


def to_doc_pos(token_idx):
    """Index in tokens.bin -> (doc_id, position in the tokenized document)."""
    w = np.searchsorted(WINDOWS.offset.values, token_idx, side="right") - 1
    body = META["max_len"] - META["prefix_len"]  # document tokens per window
    pos = WINDOWS.window.values[w] * body + (token_idx - WINDOWS.offset.values[w]) - META["prefix_len"]
    return WINDOWS.doc_id.values[w], pos


def write_layer(layer):
    # best K over all ranks
    vals = torch.cat([part["vals"][layer].float() for part in PARTS], dim=-1)  # [3, d_ff, world * K]
    ids = torch.cat([part["ids"][layer].long() for part in PARTS], dim=-1)
    vals, order = vals.topk(K, dim=-1)
    vals, ids = vals.numpy(), ids.gather(-1, order).numpy()  # [3, d_ff, K]

    os.makedirs(f"{ARGS.out_dir}/layer_{layer:02d}", exist_ok=True)
    for i, name in enumerate(LISTS):
        records = np.empty(vals[i].shape, dtype=RECORD)
        records["doc_id"], records["pos"], records["activation"] = -1, -1, np.nan
        found = np.isfinite(vals[i])
        doc_id, pos = to_doc_pos(ids[i][found])
        records["doc_id"][found], records["pos"][found] = doc_id, pos
        records["activation"][found] = SIGN[name] * vals[i][found]
        np.save(f"{ARGS.out_dir}/layer_{layer:02d}/{name}.npy", records)
    return layer


def main():
    global ARGS, PARTS, WINDOWS, K, META  # read by the forked workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--partials_dir", required=True, help="--out_dir of collect.py")
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow_incomplete", action="store_true", help="export even if some rank has not finished")
    ARGS = ap.parse_args()

    PARTS = [torch.load(p, mmap=True) for p in sorted(glob.glob(f"{ARGS.partials_dir}/rank_*.pt"))]
    if not PARTS or len(PARTS) != PARTS[0]["world"]:
        raise SystemExit(f"expected {PARTS[0]['world'] if PARTS else '?'} rank files, found {len(PARTS)}")
    complete = all(part["done"] for part in PARTS)
    if not complete and not ARGS.allow_incomplete:
        raise SystemExit("some ranks are not done (use --allow_incomplete to export anyway)")
    K = PARTS[0]["k"]
    META = json.load(open(f"{ARGS.corpus_dir}/meta.json"))
    WINDOWS = pd.read_parquet(f"{ARGS.corpus_dir}/windows.parquet", columns=["doc_id", "window", "offset"])
    n_layers = PARTS[0]["vals"].shape[0]

    os.makedirs(ARGS.out_dir, exist_ok=True)
    index = {
        "model_path": PARTS[0]["model_path"], "corpus_dir": os.path.abspath(ARGS.corpus_dir),
        "activation": "act_fn(gate_proj(x)) * up_proj(x), the input of mlp.down_proj",
        "lists": {"top_pos_pos": "gate>0, up>0; highest first", "top_neg_neg": "gate<0, up<0; highest first",
                  "bottom_mixed": "opposite signs; most negative first"},
        "files": "layer_XX/<list>.npy: [d_ff, K] records (doc_id, pos, activation), column = rank; "
                 "pos = token index in the tokenized document (no BOS); empty slot: doc_id -1, activation nan",
        "k": K, "max_len": META["max_len"], "n_layers": n_layers,
        "tokens_scanned": sum(part["tokens_done"] for part in PARTS), "complete": complete,
    }
    json.dump(index, open(f"{ARGS.out_dir}/index.json", "w"), indent=2)

    with multiprocessing.get_context("fork").Pool(ARGS.workers) as pool:
        for layer in pool.imap_unordered(write_layer, range(n_layers)):
            print(f"layer {layer:02d} written", flush=True)


if __name__ == "__main__":
    main()
