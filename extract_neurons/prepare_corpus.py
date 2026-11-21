"""Download, tokenize and store the corpus (needs internet: login node or lrd_all_serial).

Each document is tokenized in full and split into consecutive windows of max_len
tokens, so no token is lost. If the tokenizer has a BOS (Gemma) every window is
[BOS] + max_len-1 tokens; otherwise (Qwen) it is max_len tokens with no prefix.
The first token of a window (BOS, or the first text token) is never ranked: it has
no context and acts as attention sink. Output in --out_dir:
  tokens.bin       uint32 token ids of all windows, concatenated
  mask.bin         uint8 per token: 1 = rankable, 0 = first token of a window or chat-template token
  windows.parquet  one row per window: doc_id, window, source, source_row, source_id,
                   offset (into tokens.bin), length
  meta.json        max_len, prefix_len (1 with BOS, else 0), token/document counts

A firing is linked as (doc_id, pos), pos = index of the token in the tokenized
document (no BOS). With body = max_len - prefix_len, it sits in window pos // body,
at position pos % body + prefix_len.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

from model_tokenizer import chat_text, load_tokenizer


def encode(row, src, tok):
    """Token ids of one document (no BOS) and, per token, whether it may be ranked."""
    if src.get("format") != "chat":
        ids = tok(row[src["field"]] or "", add_special_tokens=False)["input_ids"]
        return ids, [True] * len(ids)
    # Chat: only tokens lying entirely inside a message's content are rankable,
    # template tokens (turn markers, roles, newlines, Qwen's empty <think></think>) are not.
    messages = row[src["field"]]
    text = chat_text(tok, messages).removeprefix(tok.bos_token or "")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    in_content = np.zeros(len(text) + 1, dtype=bool)
    start = 0
    for m in messages:
        content = m["content"].strip()  # templates may trim message contents
        start = text.find(content, start)
        if start < 0:
            raise ValueError("message content not found in the chat text")
        in_content[start: start + len(content)] = True
        start += len(content)
    keep = [b > a and in_content[a:b].all() for a, b in enc["offset_mapping"]]
    return enc["input_ids"], keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True, help="local path of the model (its tokenizer is used)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tok = load_tokenizer(args.tokenizer)
    prefix = [] if tok.bos_token_id is None else [tok.bos_token_id]
    body = cfg["max_len"] - len(prefix)  # document tokens per window
    os.makedirs(args.out_dir, exist_ok=True)

    windows, counts, offset, doc_id = [], {}, 0, 0
    with open(f"{args.out_dir}/tokens.bin", "wb") as f_tok, open(f"{args.out_dir}/mask.bin", "wb") as f_mask:
        for src in cfg["sources"]:
            ds = load_dataset(src["path"], name=src.get("name"), data_dir=src.get("data_dir"),
                              split=src.get("split", "train"), streaming=True)
            n_tokens = n_docs = 0
            for row_idx, row in enumerate(ds):
                if n_tokens >= src["tokens"]:
                    break
                try:
                    ids, keep = encode(row, src, tok)
                except Exception:  # e.g. a chat with non-alternating roles
                    continue
                if len(ids) + 1 < cfg.get("min_tokens", 1):
                    continue
                for w, start in enumerate(range(0, len(ids), body)):
                    if n_tokens >= src["tokens"]:
                        break
                    chunk = prefix + ids[start: start + body]
                    rankable = [False] * len(prefix) + keep[start: start + body]
                    rankable[0] = False  # first token of the window
                    np.array(chunk, dtype=np.uint32).tofile(f_tok)
                    np.array(rankable, dtype=np.uint8).tofile(f_mask)
                    windows.append(dict(doc_id=doc_id, window=w, source=src["label"], source_row=row_idx,
                                        source_id=str(row.get(src.get("id_field"), "")), offset=offset, length=len(chunk)))
                    offset += len(chunk)
                    n_tokens += len(chunk)
                doc_id += 1
                n_docs += 1
            counts[src["label"]] = {"tokens": n_tokens, "docs": n_docs}
            print(f"[{src['label']}] {n_tokens:,} tokens, {n_docs:,} docs", flush=True)

    pd.DataFrame(windows).to_parquet(f"{args.out_dir}/windows.parquet", index=False)
    meta = {"max_len": cfg["max_len"], "prefix_len": len(prefix), "tokenizer": args.tokenizer, "n_tokens": offset,
            "n_docs": doc_id, "n_windows": len(windows), "sources": counts}
    json.dump(meta, open(f"{args.out_dir}/meta.json", "w"), indent=2)
    print(f"total: {offset:,} tokens, {doc_id:,} docs, {len(windows):,} windows", flush=True)
    os._exit(0)  # unfinished streaming iterators keep threads alive and would block a normal exit


if __name__ == "__main__":
    main()
