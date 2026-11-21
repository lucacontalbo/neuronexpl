"""Tasks. A task is a class built from the command-line args, with:
  labels            list of label strings; each must be a single token for the classifier's tokenizer
  description       one sentence describing the task, shown to the perturber
  data_path         default dataset path
  load(path)        DataFrame with at least the columns "id" and "text" (the part the perturber edits)
  messages(row, text)   chat messages for the classifier; `text` must appear in them verbatim, and its
                        last occurrence must be the text itself (put it after the other fields of the input)
To add a task: write tasks/<name>.py and register it below.
"""
from tasks.asap2 import ASAP2
from tasks.biggen import BiGGen
from tasks.judgebench import JudgeBench

TASKS = {"asap2": ASAP2, "biggen": BiGGen, "judgebench": JudgeBench}
