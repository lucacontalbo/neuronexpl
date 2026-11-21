"""BiGGen Bench (https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results), human_eval split:
a response to an instruction is graded from 1 to 5 with the instance-specific rubric (Prometheus absolute grading,
with the response moved to the end, so that its last occurrence in the prompt is the response itself).
The perturber edits the response. Data: python data/prepare_biggen.py
"""
import pandas as pd

PROMPT = """###Task Description:
An instruction (might include an Input inside it), a reference answer that gets a score of 5, a score rubric representing an evaluation criteria and a response to evaluate are given.
1. Assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. Give a score that is an integer between 1 and 5. You should refer to the score rubric.

###The instruction to evaluate:
{instruction}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Response to evaluate:
{response}

Answer with the score only: a single number from 1 to 5."""


class BiGGen:
    labels = ["1", "2", "3", "4", "5"]
    description = ("rubric-based grading: a judge reads an instruction, a response to it, a reference answer and a "
                   "score rubric, and grades the response from 1 (lowest) to 5 (highest) according to the rubric")
    data_path = "data/biggen/human_eval.csv"

    def __init__(self, args):
        pass

    def load(self, path):
        return pd.read_csv(path, keep_default_na=False).rename(columns={"uuid": "id", "response": "text"})

    def messages(self, row, text):
        instruction = f"{row['system_prompt']}\n\n{row['input']}" if row["system_prompt"] else row["input"]
        prompt = PROMPT.format(instruction=instruction, response=text, reference=row["reference_answer"],
                               rubric=row["rubric"])
        return [{"role": "user", "content": prompt}]
