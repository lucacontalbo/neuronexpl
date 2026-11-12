"""ASAP 2.0 automatic essay scoring (https://github.com/scrosseye/ASAP_2.0): holistic score from 1 to 6."""
import os

import pandas as pd

RUBRIC = """Assign a holistic score based on the rubric below, using a grading scale between 1 (minimum) and 6 (maximum).
SCORE OF 6: An essay in this category demonstrates clear and consistent mastery, although it may have a few minor errors. A typical essay effectively and insightfully develops a point of view on the issue and demonstrates outstanding critical thinking; the essay uses clearly appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is well organized and clearly focused, demonstrating clear coherence and smooth progression of ideas; the essay exhibits skillful use of language, using a varied, accurate, and apt vocabulary and demonstrates meaningful variety in sentence structure; the essay is free of most errors in grammar, usage, and mechanics.
SCORE OF 5: An essay in this category demonstrates reasonably consistent mastery, although it will have occasional errors or lapses in quality. A typical essay effectively develops a point of view on the issue and demonstrates strong critical thinking; the essay generally using appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is well organized and focused, demonstrating coherence and progression of ideas; the essay exhibits facility in the use of language, using appropriate vocabulary demonstrates variety in sentence structure; the essay is generally free of most errors in grammar, usage, and mechanics.
SCORE OF 4: An essay in this category demonstrates adequate mastery, although it will have lapses in quality. A typical essay develops a point of view on the issue and demonstrates competent critical thinking; the essay using adequate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is generally organized and focused, demonstrating some coherence and progression of ideas exhibits adequate; the essay may demonstrate inconsistent facility in the use of language, using generally appropriate vocabulary demonstrates some variety in sentence structure; the essay may have some errors in grammar, usage, and mechanics.
SCORE OF 3: An essay in this category demonstrates developing mastery, and is marked by ONE OR MORE of the following weaknesses: develops a point of view on the issue, demonstrating some critical thinking, but may do so inconsistently or use inadequate examples, reasons, or other evidence taken from the source texts to support its position; the essay is limited in its organization or focus, or may demonstrate some lapses in coherence or progression of ideas displays; the essay may demonstrate facility in the use of language, but sometimes uses weak vocabulary or inappropriate word choice and/or lacks variety or demonstrates problems in sentence structure; the essay may contain an accumulation of errors in grammar, usage, and mechanics.
SCORE OF 2: An essay in this category demonstrates little mastery, and is flawed by ONE OR MORE of the following weaknesses: develops a point of view on the issue that is vague or seriously limited, and demonstrates weak critical thinking; the essay provides inappropriate or insufficient examples, reasons, or other evidence taken from the source text to support its position; the essay is poorly organized and/or focused, or demonstrates serious problems with coherence or progression of ideas; the essay displays very little facility in the use of language, using very limited vocabulary or incorrect word choice and/or demonstrates frequent problems in sentence structure; the essay contains errors in grammar, usage, and mechanics so serious that meaning is somewhat obscured.
SCORE OF 1: An essay in this category demonstrates very little or no mastery, and is severely flawed by ONE OR MORE of the following weaknesses: develops no viable point of view on the issue, or provides little or no evidence to support its position; the essay is disorganized or unfocused, resulting in a disjointed or incoherent essay; the essay displays fundamental errors in vocabulary and/or demonstrates severe flaws in sentence structure; the essay contains pervasive errors in grammar, usage, or mechanics that persistently interfere with meaning."""

# prompt_name in the csv -> plain-text file of its source text(s) in --source_texts_dir
SOURCE_FILES = {
    "Facial action coding system": "IN1_facial_action_coding_system.txt",
    "Driverless cars": "IN2_driverless_cars.txt",
    "Exploring Venus": "IN3_exploring_venus.txt",
    '"A Cowboy Who Rode the Waves"': "IN5_a_cowboy_who_rode_the_waves.txt",
    "The Face on Mars": "IN6_the_face_on_mars.txt",
}


class ASAP2:
    labels = ["1", "2", "3", "4", "5", "6"]
    description = "automatic essay scoring: a student essay written for an assignment receives a holistic score from 1 (lowest) to 6 (highest) according to a rubric"
    data_path = "data/asap2/ASAP_2_Final_github_test.csv"

    def __init__(self, args):
        self.source_texts_dir = args.source_texts_dir if args.include_source_text else None

    def load(self, path):
        return pd.read_csv(path).rename(columns={"essay_id": "id", "full_text": "text"})

    def messages(self, row, text):
        source = ""
        if self.source_texts_dir:
            with open(os.path.join(self.source_texts_dir, SOURCE_FILES[row["prompt_name"]])) as f:
                source = f"Source text(s):\n{f.read().strip()}\n\n"
        prompt = (f"Grade the following student essay.\n\n"
                  f"Assignment:\n{row['assignment']}\n\n"
                  f"{source}"
                  f"Rubric:\n{RUBRIC}\n\n"
                  f"Essay:\n{text}\n\n"
                  f"Answer with the score only: a single number from 1 to 6.")
        return [{"role": "user", "content": prompt}]
