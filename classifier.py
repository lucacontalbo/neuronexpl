"""Defending LLM. Its label is the most likely label token right after the chat prompt (greedy, temperature 0).
Neurons are the MLP units of every decoder layer: the input of mlp.down_proj, act(gate_proj(x)) * up_proj(x),
the same quantity mapped by extract_neurons/collect.py."""
from dataclasses import dataclass
from functools import partial

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


@dataclass
class State:
    """One forward pass on one text."""
    input_ids: torch.Tensor  # [1, seq]
    positions: torch.Tensor  # [P] indices of the tokens of the editable text
    logits: torch.Tensor     # [n_labels], attached to the autograd graph
    acts: list               # per layer [1, seq, d_ff], attached to the autograd graph
    pred: str

    def activations(self):
        """[L, P, d_ff] activations at the editable text tokens."""
        return torch.stack([a[0, self.positions] for a in self.acts]).detach().float()


class Classifier:
    def __init__(self, model_path, labels, dtype=torch.float32, device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.layers = self.model.get_decoder().layers
        self.n_layers, self.d_ff = len(self.layers), self.layers[0].mlp.down_proj.in_features
        self.labels = labels
        self.label_ids = [self._token_id(label) for label in labels]

        self._acts = [None] * self.n_layers
        self._patch = []  # [(layer, positions, neuron, value)] activations to overwrite in the next forward
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(partial(self._hook, i))
        # the weights are frozen: make the embeddings require grad so that activations get gradients
        self.model.get_input_embeddings().register_forward_hook(
            lambda module, inputs, out: out.requires_grad_(True) if torch.is_grad_enabled() else out)

    def _token_id(self, label):
        ids = self.tok.encode(label, add_special_tokens=False)
        assert len(ids) == 1, f"label {label!r} is not a single token"
        return ids[0]

    def _hook(self, layer, module, args):
        h = args[0]
        for l, positions, neuron, value in self._patch:
            if l == layer:
                h = h.clone()
                h[0, positions, neuron] = value
        self._acts[layer] = h
        return (h,)

    def encode(self, messages, text):
        """Token ids of the chat prompt, and the positions of the tokens lying inside `text`."""
        chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = chat.find(text)
        assert start >= 0, "the text is not in the prompt"
        enc = self.tok(chat, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        a, b = enc["offset_mapping"][0].T
        positions = ((a >= start) & (b <= start + len(text)) & (b > a)).nonzero()[:, 0]
        return enc["input_ids"].to(self.device), positions.to(self.device)

    def run(self, messages, text):
        input_ids, positions = self.encode(messages, text)
        with torch.enable_grad():
            logits = self.model(input_ids=input_ids, logits_to_keep=1).logits[0, -1, self.label_ids].float()
        return State(input_ids, positions, logits, list(self._acts), self.labels[logits.argmax()])

    def margin(self, state, target, current):
        """J = log P(target) - log P(current), i.e. the difference of their logits."""
        t, c = self.labels.index(target), self.labels.index(current)
        return (state.logits[t] - state.logits[c]).item()

    def gradients(self, state, target, current):
        """[L, P, d_ff] dJ/da at the editable text tokens."""
        t, c = self.labels.index(target), self.labels.index(current)
        grads = torch.autograd.grad(state.logits[t] - state.logits[c], state.acts, retain_graph=True)
        return torch.stack([g[0, state.positions] for g in grads]).float()

    @torch.no_grad()
    def patched_margin(self, state, target, current, patch):
        """J with some activations overwritten; patch = [(layer, positions (indices into state.positions), neuron, value)]."""
        t, c = self.labels.index(target), self.labels.index(current)
        self._patch = [(layer, state.positions[p], neuron, value) for layer, p, neuron, value in patch]
        try:
            logits = self.model(input_ids=state.input_ids, logits_to_keep=1).logits[0, -1, self.label_ids].float()
        finally:
            self._patch = []
        return (logits[t] - logits[c]).item()
