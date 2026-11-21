"""Neuron search.

For neuron n of layer l at text token t, with activation a and gradient g = dJ/da (J = log P(target) - log P(current)),
the estimated gain of moving the activation to a naturally observed value a* (from the neuron map) is
    S = g * (a* - a)
Directions:
    "+" raise the neuron: a* = mean of its top `a_star_n` activations in top_pos_pos
    "-" lower the neuron: a* = mean of its `a_star_n` most negative activations in bottom_mixed (only with --use_negative)
S is pooled over the text tokens (max, mean or mean of the top-p), the best neurons over all layers are taken and,
unless disabled, re-ranked by the real change of J when the neuron is patched to a* (causal check).

Strategies: "gradient" (above), "negative" (ablation: the neurons with the most negative S, i.e. the ones
predicted to push away from the target), "random" (ablation: random neurons, raised).
"""
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from extract_neurons.show import load_layer, load_list

LISTS = {"+": "top_pos_pos", "-": "bottom_mixed"}  # where a* and the contexts of each direction come from


@dataclass(frozen=True)
class Neuron:
    layer: int
    index: int
    direction: str  # "+" raise, "-" lower
    a_star: float

    @property
    def key(self):
        return f"{self.layer}:{self.index}"


def _a_star(map_dir, n_layers, n):
    """[2, L, d_ff]: for each direction, the mean of the n strongest activations of every neuron (nan if empty)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty list: the mean of nothing is nan
        return torch.tensor(np.array([
            [np.nanmean(load_layer(map_dir, layer, name)["activation"][:, :n], axis=1, dtype=np.float64)
             for layer in range(n_layers)]
            for name in LISTS.values()
        ]), dtype=torch.float32)


class NeuronMap:
    """a* of every neuron and the top contexts of a neuron, from the output of extract_neurons/."""

    def __init__(self, map_dir, corpus_dir, tokenizer, n_layers, d_ff, a_star_n=10):
        self.map_dir, self.tok = map_dir, tokenizer
        cache = f"{map_dir}/a_star_top{a_star_n}.pt"
        if not os.path.exists(cache):
            print(f"computing a* from {map_dir} (once, cached in {cache})", flush=True)
            torch.save(_a_star(map_dir, n_layers, a_star_n), f"{cache}.{os.getpid()}.tmp")
            os.replace(f"{cache}.{os.getpid()}.tmp", cache)  # atomic: parallel jobs may compute it at the same time
        a_star = torch.load(cache)  # [2, L, d_ff]
        assert a_star.shape[1:] == (n_layers, d_ff), "the neuron map does not match the model"
        self.a_star = dict(zip(LISTS, a_star))

        # corpus, to decode the contexts (as in extract_neurons/show.py)
        meta = json.load(open(f"{corpus_dir}/meta.json"))
        self.prefix = meta["prefix_len"]
        self.body = meta["max_len"] - self.prefix
        self.windows = pd.read_parquet(f"{corpus_dir}/windows.parquet").set_index(["doc_id", "window"])
        self.tokens = np.memmap(f"{corpus_dir}/tokens.bin", dtype=np.uint32, mode="r")

    def contexts(self, neuron, n=10, before=120, after=40):
        """The n strongest firings of the neuron in its direction, firing token in [[ ]]."""
        out = []
        for doc, pos, _ in load_list(self.map_dir, neuron.layer, neuron.index, LISTS[neuron.direction])[:n]:
            w, p = pos // self.body, pos % self.body + self.prefix
            win = self.windows.loc[(doc, w)]
            t = self.tokens[win.offset: win.offset + win.length].tolist()
            out.append(self.tok.decode(t[max(0, p - before): p]) + f"[[{self.tok.decode(t[p: p + 1])}]]" + self.tok.decode(t[p + 1: p + 1 + after]))
        return out


class NeuronSearch:
    def __init__(self, classifier, neuron_map, strategy="gradient", use_negative=False, pooling="max", pool_topk=5,
                 causal_candidates=50, seed=0, candidates_per_layer=2):
        self.clf, self.strategy, self.pooling, self.pool_topk = classifier, strategy, pooling, pool_topk
        self.causal_candidates = causal_candidates  # 0 = no causal check, rank by S only
        self.candidates_per_layer = candidates_per_layer  # >0: the causal check gets the top-N by S of every layer
        self.directions = ["+", "-"] if use_negative else ["+"]
        self.a_star = {d: v.to(classifier.device) for d, v in neuron_map.a_star.items()}
        self.rng = torch.Generator().manual_seed(seed)

    def select(self, state, target, current, k, used):
        """k new neurons (not in `used`, a set of (layer, index)) to push the classifier from `current` to `target`."""
        if self.strategy == "random":
            return self._random(k, used)
        orient = 1 if self.strategy == "gradient" else -1  # "negative": rank by -S
        grads = self.clf.gradients(state, target, current)  # per layer [P, d_ff], on the GPU of the layer

        # pooled S of every neuron, one layer at a time (a [L, P, d_ff] tensor would not fit for large models)
        pooled = []  # per layer [n_directions, d_ff]
        for layer, g in enumerate(grads):
            acts = state.layer_activations(layer)
            pooled.append(torch.stack([self._pool(self._score(orient, g, acts, self.a_star[d][layer].to(g.device)))
                                       for d in self.directions]).to(self.clf.device))
        best, best_dir = torch.stack(pooled, dim=1).max(0)  # [L, d_ff]: best direction of every neuron
        for layer, index in used:
            best[layer, index] = float("-inf")
        if self.causal_candidates and self.candidates_per_layer:
            # stratified pool: the scale of S differs by layer, so a global top-N comes from the first layers only;
            # the causal check (real dJ, same units for every layer) then decides between layers
            values, idx = best.topk(min(self.candidates_per_layer, best.shape[1]), dim=1)  # [L, N]
            flat = idx + torch.arange(best.shape[0], device=idx.device)[:, None] * best.shape[1]
            values, flat = values.flatten(), flat.flatten()
        else:
            values, flat = best.flatten().topk(self.causal_candidates or k)
        candidates = []
        for value, i in zip(values.tolist(), flat.tolist()):
            if value == float("-inf"):
                continue
            layer, index = divmod(i, best.shape[1])
            d = self.directions[best_dir[layer, index]]
            candidates.append(Neuron(layer, index, d, self.a_star[d][layer, index].item()))
        if not self.causal_candidates:
            return candidates[:k]

        # causal check: real change of J when the neuron is set to a* where it scored
        j0 = self.clf.margin(state, target, current)
        effects = []
        for n in candidates:
            s = orient * grads[n.layer][:, n.index] * (n.a_star - state.neuron_activations(n.layer, n.index))
            patch = [(n.layer, self._positions(s), n.index, n.a_star)]
            effects.append(orient * (self.clf.patched_margin(state, target, current, patch) - j0))
        ranked = sorted(zip(effects, candidates), key=lambda x: -x[0])
        helpful = [n for effect, n in ranked if effect > 0]
        return (helpful or [n for _, n in ranked])[:k]  # if no candidate helps, still take the highest effects

    def pooled_activation(self, state, neuron):
        """Activation of the neuron over the text tokens, pooled as in the search ("-": its lowest values)."""
        sign = 1 if neuron.direction == "+" else -1
        acts = state.neuron_activations(neuron.layer, neuron.index)
        return sign * self._pool(sign * acts[:, None])[0].item()

    @staticmethod
    def _score(orient, grads, acts, a_star):
        """[P, d_ff] oriented S = g * (a* - a); -inf for the neurons without a* (empty list in the map)."""
        return (orient * grads * (a_star - acts)).nan_to_num(nan=float("-inf"))

    def _pool(self, x):  # [..., P, F] -> [..., F]
        if self.pooling == "max":
            return x.max(-2).values
        if self.pooling == "mean":
            return x.mean(-2)
        return x.topk(min(self.pool_topk, x.shape[-2]), dim=-2).values.mean(-2)

    def _positions(self, s):  # [P] -> the positions that made the pooled score
        if self.pooling == "max":
            return s.argmax().view(1)
        if self.pooling == "topk":
            return s.topk(min(self.pool_topk, len(s))).indices
        return torch.arange(len(s), device=s.device)  # mean: S was averaged over all text tokens, patch them all

    def _random(self, k, used):
        ok = self.a_star["+"].isfinite().cpu()
        for layer, index in used:
            ok[layer, index] = False
        flat = ok.flatten().nonzero()[:, 0]
        pick = flat[torch.randperm(len(flat), generator=self.rng)[:k]].tolist()
        return [Neuron(i // ok.shape[1], i % ok.shape[1], "+", self.a_star["+"].flatten()[i].item()) for i in pick]
