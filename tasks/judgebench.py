"""JudgeBench (https://huggingface.co/datasets/ScalerLab/JudgeBench), gpt + claude splits: pairwise comparison of
two responses to a question, the judge answers A or B. The perturber always edits response B (target "B": make B
win; target "A": make B lose). Data: python data/prepare_judgebench.py
"""
import pandas as pd

PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Choose the assistant that follows the user's instructions and answers the user's question better, considering above all the correctness of the answer. Do not let the order in which the responses are presented, their length or the names of the assistants influence your evaluation.

[User Question]
{question}

[The Start of Assistant A's Answer]
{a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{b}
[The End of Assistant B's Answer]

Answer with a single letter: "A" if assistant A is better, "B" if assistant B is better."""


class JudgeBench:
    labels = ["A", "B"]
    description = ("pairwise response comparison: a judge reads a user question and the answers of two assistants, "
                   "A and B, and outputs which one is better (A or B); the text to edit is the answer of assistant B")
    data_path = "data/judgebench/judgebench.csv"

    def __init__(self, args):
        pass

    def load(self, path):
        return pd.read_csv(path, keep_default_na=False).rename(columns={"response_B": "text"})

    def messages(self, row, text):
        prompt = PROMPT.format(question=row["question"], a=row["response_A"], b=text)
        return [{"role": "user", "content": prompt}]
