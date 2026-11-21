"""Print the context of firings (the window the model saw; the firing token in [[ ]]).

  python show.py --corpus_dir C --tokenizer M --doc 123 --pos 45
  python show.py --corpus_dir C --tokenizer M --map_dir OUT --layer 10 --neuron 42 --list top_neg_neg -n 5
"""
import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def load_layer(map_dir, layer, name):
    """[d_ff, K] records (doc_id, pos, activation) of one list of one layer (memory-mapped: only what is used is read)."""
    return np.load(f"{map_dir}/layer_{layer:02d}/{name}.npy", mmap_mode="r")


def load_list(map_dir, layer, neuron, name):
    """[(doc_id, pos, activation)] of one list of one neuron, strongest first."""
    records = load_layer(map_dir, layer, name)[neuron]
    records = records[records["doc_id"] >= 0]  # drop the empty slots
    return list(zip(records["doc_id"].tolist(), records["pos"].tolist(), records["activation"].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--doc", type=int)
    ap.add_argument("--pos", type=int)
    ap.add_argument("--map_dir")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--neuron", type=int)
    ap.add_argument("--list", default="top_pos_pos", choices=["top_pos_pos", "top_neg_neg", "bottom_mixed"])
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--before", type=int, default=40, help="context tokens shown before the firing token")
    ap.add_argument("--after", type=int, default=10, help="tokens shown after it")
    args = ap.parse_args()

    from model_tokenizer import load_tokenizer  # here, not at the top: neuron_search.py imports this file as a module
    tok = load_tokenizer(args.tokenizer)
    meta = json.load(open(f"{args.corpus_dir}/meta.json"))
    prefix = meta["prefix_len"]  # 1 if windows start with BOS, else 0
    body = meta["max_len"] - prefix  # document tokens per window
    wins = pd.read_parquet(f"{args.corpus_dir}/windows.parquet").set_index(["doc_id", "window"])
    tokens = np.memmap(f"{args.corpus_dir}/tokens.bin", dtype=np.uint32, mode="r")

    if args.map_dir:
        firings = load_list(args.map_dir, args.layer, args.neuron, args.list)[: args.n]
    else:
        firings = [(args.doc, args.pos, None)]

    for rank, (doc, pos, act) in enumerate(firings):
        w, p = pos // body, pos % body + prefix  # window, and position inside it
        win = wins.loc[(doc, w)]
        t = tokens[win.offset: win.offset + win.length].tolist()
        context = tok.decode(t[max(0, p - args.before): p]) + f"[[{tok.decode(t[p:p + 1])}]]" + tok.decode(t[p + 1: p + 1 + args.after])
        head = f"#{rank} act={act:.4g} " if act is not None else ""
        print(f"{head}doc={doc} pos={pos} (window {w}) source={win.source} row={win.source_row} id={win.source_id}")
        print("  ..." + context.replace("\n", "\\n") + "...")


if __name__ == "__main__":
    main()
