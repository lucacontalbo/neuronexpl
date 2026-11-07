"""Merge the per-rank buffers of collect.py and write one file per neuron and list:
  <out_dir>/layer_XX/neuron_NNNNN/{top_pos_pos,top_neg_neg,bottom_mixed}.{csv,npy}
Records (rank, doc_id, pos, activation), strongest first (bottom_mixed: most negative first).
pos is the token index in the tokenized document (no BOS), so same doc_id = same input.
A list has fewer than K records if its regime fired fewer than K times in the corpus.
  csv: header "rank,doc_id,pos,activation"    npy: fields doc_id, pos, activation (rank = row)
"""
import argparse
import glob
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch

LISTS = [("top_pos_pos", 1), ("top_neg_neg", 1), ("bottom_mixed", -1)]  # (name, activation = sign * score)
NPY = np.dtype([("doc_id", "<i4"), ("pos", "<i4"), ("activation", "<f4")])


def to_links(token_idx):
    """Index in tokens.bin -> (doc_id, position in the tokenized document)."""
    w = np.searchsorted(WIN.offset.values, token_idx, side="right") - 1
    pos_in_window = token_idx - WIN.offset.values[w]
    body = META["max_len"] - META["prefix_len"]  # document tokens per window
    return WIN.doc_id.values[w], WIN.window.values[w] * body + pos_in_window - META["prefix_len"]


def write_layer(layer):
    vals = torch.cat([p["vals"][layer].float() for p in PARTS], -1)
    ids = torch.cat([p["ids"][layer].long() for p in PARTS], -1)
    vals, order = vals.topk(K, dim=-1)
    vals, ids = vals.numpy(), ids.gather(-1, order).numpy()  # [3, d_ff, K]
    for n in range(vals.shape[1]):
        out = f"{ARGS.out_dir}/layer_{layer:02d}/neuron_{n:05d}"
        os.makedirs(out, exist_ok=True)
        for i, (name, sign) in enumerate(LISTS):
            found = np.isfinite(vals[i, n])
            act = sign * vals[i, n][found]
            doc, pos = to_links(ids[i, n][found])
            if ARGS.format == "csv":
                rows = (f"{r},{d},{p},{a:.6g}\n" for r, (d, p, a) in enumerate(zip(doc.tolist(), pos.tolist(), act.tolist())))
                with open(f"{out}/{name}.csv", "w") as f:
                    f.write("rank,doc_id,pos,activation\n")
                    f.writelines(rows)
            else:
                rec = np.empty(len(act), dtype=NPY)
                rec["doc_id"], rec["pos"], rec["activation"] = doc, pos, act
                np.save(f"{out}/{name}.npy", rec)
    return layer


def main():
    global ARGS, PARTS, WIN, K, META  # shared with the forked workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--partials_dir", required=True, help="--out_dir of collect.py")
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--format", choices=["csv", "npy"], default="csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow_incomplete", action="store_true", help="export even if some rank has not finished")
    ARGS = ap.parse_args()

    PARTS = [torch.load(p, mmap=True) for p in sorted(glob.glob(f"{ARGS.partials_dir}/rank_*.pt"))]
    if not PARTS or len(PARTS) != PARTS[0]["world"]:
        raise SystemExit(f"expected {PARTS[0]['world'] if PARTS else '?'} rank files, found {len(PARTS)}")
    complete = all(p["done"] for p in PARTS)
    if not complete and not ARGS.allow_incomplete:
        raise SystemExit("some ranks are not done (use --allow_incomplete to export anyway)")
    K = PARTS[0]["k"]
    META = json.load(open(f"{ARGS.corpus_dir}/meta.json"))
    WIN = pd.read_parquet(f"{ARGS.corpus_dir}/windows.parquet", columns=["doc_id", "window", "offset"])
    n_layers = PARTS[0]["vals"].shape[0]

    os.makedirs(ARGS.out_dir, exist_ok=True)
    json.dump({
        "model_path": PARTS[0]["model_path"], "corpus_dir": os.path.abspath(ARGS.corpus_dir),
        "activation": "gelu_tanh(gate_proj(x)) * up_proj(x), the input of mlp.down_proj",
        "lists": {"top_pos_pos": "gate>0, up>0; highest first", "top_neg_neg": "gate<0, up<0; highest first",
                  "bottom_mixed": "opposite signs; most negative first"},
        "record": "rank, doc_id, pos, activation; pos = token index in the tokenized document (no BOS)",
        "k": K, "max_len": META["max_len"], "n_layers": n_layers, "format": ARGS.format,
        "tokens_scanned": sum(p["tokens_done"] for p in PARTS), "complete": complete,
    }, open(f"{ARGS.out_dir}/index.json", "w"), indent=2)

    with multiprocessing.get_context("fork").Pool(ARGS.workers) as pool:
        for layer in pool.imap_unordered(write_layer, range(n_layers)):
            print(f"layer {layer:02d} written", flush=True)


if __name__ == "__main__":
    main()
