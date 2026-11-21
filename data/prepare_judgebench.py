"""Download JudgeBench (evaluation-only benchmark; gpt split 350 pairs + claude split 270 pairs) to
data/judgebench/judgebench.csv. Needs internet (run on a login node)."""
import os

import pandas as pd

URL = "https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/main/data/{split}-00000-of-00001.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgebench", "judgebench.csv")

df = pd.concat([pd.read_json(URL.format(split=s), lines=True).assign(split=s) for s in ["gpt", "claude"]])
df["id"] = df.split + "_" + df.pair_id.astype(str)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df[["id", "split", "pair_id", "original_id", "source", "response_model", "question", "response_A", "response_B",
    "label"]].to_csv(OUT, index=False)
print(f"{len(df)} pairs -> {OUT}")
