"""Download the BiGGen Bench human_eval split (test data, English, 2780 rows) to data/biggen/human_eval.csv.
Needs internet (run on a login node). Rows with an empty response are dropped (no text to perturb)."""
import os

import pandas as pd

URL = "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results/resolve/main/data/human_eval-00000-of-00001.parquet"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biggen", "human_eval.csv")


def rubric(r):
    return f"[{r['criteria']}]\n" + "\n".join(f"Score {i}: {r[f'score{i}_description']}" for i in range(1, 6))


df = pd.read_parquet(URL)
assert not df.used_for_training.astype(str).eq("True").any()
df["rubric"] = df.score_rubric.map(rubric)
empty = df.response.str.strip() == ""
print(f"dropped {empty.sum()} rows with an empty response")
df = df[~empty]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.rename(columns={"id": "instance_id"})[["uuid", "instance_id", "capability", "task", "model_name", "system_prompt",
                                          "input", "reference_answer", "rubric", "response", "human_score"]].to_csv(OUT, index=False)
print(f"{len(df)} rows -> {OUT}")
