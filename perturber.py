"""Perturber LLM (OpenAI API; the key is read from .env / OPENAI_API_KEY)."""
import re

from dotenv import load_dotenv
from openai import OpenAI

FIXED_TEMPERATURE = ("gpt-5", "o1", "o3", "o4")  # models that only accept temperature = 1

PROMPT = """You are helping to test the robustness of a text classifier.

Classifier task: {task}
The classifier currently assigns the label "{current}" to the text below. Your goal is to edit the text so that the classifier assigns the label "{target}" instead.

You may ONLY make the kind of change described by the patterns below. Each pattern belongs to one internal unit (a "neuron") of the classifier, and is shown through the excerpts of a large corpus where the unit reaches the activation we want; in each excerpt the token where it fires is marked as [[token]]. First work out what the excerpts of each pattern have in common (e.g. a word, a topic, a style, a grammatical construction, a punctuation habit).

{patterns}

Rules:
1. Edit the text so that it shows more of these patterns, in a way that could move the classifier toward "{target}".
2. Use ONLY these patterns: make no other change (no other corrections, additions, removals or rewordings).
3. Apart from these changes, the text must stay as close as possible to the original: same content, structure, length and style.

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
        return match.group(1) if match else reply.strip()
