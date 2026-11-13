"""Perturber LLM (OpenAI API; the key is read from .env / OPENAI_API_KEY)."""
import re

from dotenv import load_dotenv
from openai import OpenAI

FIXED_TEMPERATURE = ("gpt-5", "o1", "o3", "o4")  # models that only accept temperature = 1

PROMPT = """You are helping to test the robustness of a text classifier.

Classifier task: {task}
The classifier currently assigns the label "{current}" to the text below. Your goal is to edit the text so that the classifier assigns the label "{target}" instead.

You may ONLY make the kind of change described by the patterns below. Each pattern belongs to one internal unit (a "neuron") of the classifier, and is shown through excerpts of a large corpus where the unit reaches the activation we want.

How to read the excerpts:
- In each excerpt, the single token where the unit fires is wrapped in [[ and ]]. These brackets are ONLY a marker added for you: they are not part of the excerpt, and they must NEVER appear in the edited text.
- The classifier reads text left to right, so the unit firing at the marked token is caused by that token together with the words right BEFORE it. The words after the marked token have no influence on it.

For each pattern, before editing:
1. Look at the marked tokens: is it the same word, a family of words (e.g. synonyms, a part of speech), a punctuation mark, a suffix, a number?
2. Look at what comes right before the marked tokens: is there a recurring construction, phrase, topic or register that leads to them?
3. From 1 and 2, state the trigger as precisely as possible: "token X right after a context like Y".
4. Find one or a few places in the text where this trigger can be inserted or reproduced so that it reads naturally, fits the meaning of the sentence and the voice of the writer.

{patterns}

Rules:
1. Integrate the trigger of each pattern into the text, with small local edits (a word, a short phrase or at most a short clause), placed where they fit naturally. The edited sentences must read as if the writer had written them.
2. Use ONLY these patterns: make no other change. Do not fix spelling, grammar or punctuation, do not reword, reorder, add or remove anything else.
3. Everything outside your edits must stay exactly identical to the original, character by character (including its errors, line breaks and formatting).
4. Never write the [[ ]] markers or any other annotation in the text.

Return only the edited text, between <text> and </text>.

Text:
<text>
{text}
</text>"""

DIRECTION = {"+": "excerpts where the unit is most strongly activated", "-": "excerpts where the unit is most strongly negative"}


class Perturber:
    def __init__(self, model, base_url=None):
        """base_url: an OpenAI-compatible server (e.g. vLLM, "http://127.0.0.1:8000/v1"); None = OpenAI API."""
        load_dotenv()
        self.client = OpenAI(base_url=base_url, api_key="EMPTY") if base_url else OpenAI()
        self.model = model
        self.temperature = 1.0 if model.startswith(FIXED_TEMPERATURE) else 0.5

    def __call__(self, text, task, current, target, neurons):
        """neurons: [(Neuron, [context strings])]. Returns the edited text."""
        patterns = "\n\n".join(
            f"Pattern {i} ({DIRECTION[n.direction]}):\n" + "\n".join(f"  - ...{c.replace(chr(10), ' ')}..." for c in contexts)
            for i, (n, contexts) in enumerate(neurons, 1))
        prompt = PROMPT.format(task=task, current=current, target=target, patterns=patterns, text=text)
        reply = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        if not reply:  # e.g. a reasoning model that spent all its tokens thinking
            print("  perturber returned no text: the text is left unchanged", flush=True)
            return text
        match = re.search(r"<text>\s*(.*?)\s*</text>", reply, re.S)
        return strip_markers(match.group(1) if match else reply.strip(), text)


def strip_markers(edited, original):
    """Remove the [[ ]] firing markers of the excerpts if the perturber copied them: [[word]] -> word, then any
    leftover [[ or ]] that the original text did not contain."""
    edited = re.sub(r"\[\[(.*?)\]\]", r"\1", edited, flags=re.S)
    for marker in ("[[", "]]"):
        if marker not in original:
            edited = edited.replace(marker, "")
    return edited
