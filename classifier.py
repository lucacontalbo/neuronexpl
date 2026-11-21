"""Defending LLM. Its label is the most likely label token right after the chat prompt (greedy, temperature 0).
Neurons are the MLP units of every decoder layer: the input of mlp.down_proj, act(gate_proj(x)) * up_proj(x),
the same quantity mapped by extract_neurons/collect.py.

Models that do not fit on one GPU (e.g. 24-27B, with gradients) are split over `gpus` GPUs: each layer's
activations and gradients stay on the GPU of that layer. The label logits are always computed in float32 from the
last hidden state, so that a bfloat16 model still gives fine-grained margins J.
"""
import os
from dataclasses import dataclass
from functools import partial

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


@dataclass
class State:
    """One forward pass on one text."""
    input_ids: torch.Tensor  # [1, seq]
    positions: torch.Tensor  # [P] indices of the tokens of the editable text
    logits: torch.Tensor     # [n_labels] float32, attached to the autograd graph
    acts: list               # per layer [1, seq, d_ff] (on the GPU of the layer), attached to the autograd graph
    pred: str

    def layer_activations(self, layer):
        """[P, d_ff] float32 activations of one layer at the editable text tokens (on the GPU of the layer)."""
        acts = self.acts[layer]
        return acts[0, self.positions.to(acts.device)].detach().float()

    def neuron_activations(self, layer, index):
        """[P] float32 activations of one neuron at the editable text tokens."""
        acts = self.acts[layer]
        return acts[0, self.positions.to(acts.device), index].detach().float()


def load_model(model_path, dtype, gpus):
    """The model on `gpus` GPUs (its layers split evenly over them if more than one), or on the cpu if there is none."""
    is_multimodal = hasattr(AutoConfig.from_pretrained(model_path), "vision_config")
    model_class = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    if not torch.cuda.is_available():
        return model_class.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
    if gpus == 1:
        return model_class.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").to("cuda:0")
    # cap every GPU at ~1/n of the weights, so that device_map="auto" spreads the layers evenly
    assert gpus <= torch.cuda.device_count(), f"--gpus {gpus} but {torch.cuda.device_count()} GPUs visible"
    weights = sum(os.path.getsize(f"{model_path}/{f}") for f in os.listdir(model_path) if f.endswith(".safetensors"))
    max_memory = {gpu: int(1.15 * weights / gpus) for gpu in range(gpus)}
    return model_class.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa",
                                       device_map="auto", max_memory=max_memory)


class Classifier:
    def __init__(self, model_path, labels, dtype=torch.float32, gpus=1):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = load_model(model_path, dtype, gpus).eval()
        self.model.requires_grad_(False)
        self.decoder = self.model.get_decoder()
        self.device = self.decoder.get_input_embeddings().weight.device  # where the inputs go
        self.layers = self.decoder.layers
        self.n_layers, self.d_ff = len(self.layers), self.layers[0].mlp.down_proj.in_features
        self.labels = labels

        # float32 rows of the output layer for the label tokens (and Gemma's logit softcapping, if any)
        label_ids = [self._token_id(label) for label in labels]
        head = self.model.get_output_embeddings().weight
        self.label_head = head[label_ids].detach().float()
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.softcap = getattr(text_config, "final_logit_softcapping", None)

        self._acts = [None] * self.n_layers
        self._patch = []  # [(layer, positions, neuron, value)] activations to overwrite in the next forward
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(partial(self._hook, i))
        # the weights are frozen: make the embeddings require grad so that activations get gradients
        self.decoder.get_input_embeddings().register_forward_hook(
            lambda module, inputs, out: out.requires_grad_(True) if torch.is_grad_enabled() else out)

    def _token_id(self, label):
        ids = self.tok.encode(label, add_special_tokens=False)
        assert len(ids) == 1, f"label {label!r} is not a single token"
        return ids[0]

    def _hook(self, layer, module, args):
        """Record the activations of the layer, overwriting the patched ones."""
        h = args[0]
        for l, positions, neuron, value in self._patch:
            if l == layer:
                h = h.clone()
                h[0, positions.to(h.device), neuron] = value
        self._acts[layer] = h
        return (h,)

    def _label_logits(self, input_ids):
        """[n_labels] float32 logits of the label tokens at the last position."""
        hidden = self.decoder(input_ids=input_ids, use_cache=False).last_hidden_state[0, -1]
        logits = self.label_head @ hidden.float().to(self.label_head.device)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def encode(self, messages, text):
        """Token ids of the chat prompt, and the positions of the tokens lying inside `text`."""
        chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = chat.rfind(text)  # last occurrence: the tasks put the edited text after the rest of the input
        assert start >= 0, "the text is not in the prompt"
        enc = self.tok(chat, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        a, b = enc["offset_mapping"][0].T
        positions = ((a >= start) & (b <= start + len(text)) & (b > a)).nonzero()[:, 0]
        return enc["input_ids"].to(self.device), positions.to(self.device)

    def run(self, messages, text):
        input_ids, positions = self.encode(messages, text)
        with torch.enable_grad():
            logits = self._label_logits(input_ids)
        return State(input_ids, positions, logits, list(self._acts), self.labels[logits.argmax()])

    def margin(self, state, target, current):
        """J = log P(target) - log P(current), i.e. the difference of their logits."""
        t, c = self.labels.index(target), self.labels.index(current)
        return (state.logits[t] - state.logits[c]).item()

    def gradients(self, state, target, current):
        """Per layer, [P, d_ff] float32 dJ/da at the editable text tokens (on the GPU of the layer)."""
        t, c = self.labels.index(target), self.labels.index(current)
        grads = torch.autograd.grad(state.logits[t] - state.logits[c], state.acts, retain_graph=True)
        return [g[0, state.positions.to(g.device)].float() for g in grads]

    @torch.no_grad()
    def patched_margin(self, state, target, current, patch):
        """J with some activations overwritten; patch = [(layer, positions (indices into state.positions), neuron, value)]."""
        t, c = self.labels.index(target), self.labels.index(current)
        self._patch = [(layer, state.positions[p.to(state.positions.device)], neuron, value) for layer, p, neuron, value in patch]
        try:
            logits = self._label_logits(state.input_ids)
        finally:
            self._patch = []
        return (logits[t] - logits[c]).item()
