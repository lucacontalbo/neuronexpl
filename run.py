"""Push a classifier toward a target label by editing its input with an LLM, guided by the classifier's neurons.

For every sample the classifier predicts a label (samples already predicted as the target are skipped). Then, up to
--n_iter times: pick --k new neurons (neuron_search.py), show their top corpus contexts to the perturber LLM, which
edits the text, and classify the edited text again; stop as soon as the target label is reached.
The ground truth is never used. One csv row per sample, appended as soon as the sample is done (rerun = resume).

  python run.py --task asap2 --model Qwen/Qwen3.5-0.8B --map_dir MAP --corpus_dir CORPUS --target_label 6 --k 3 --n_iter 5 --output out.csv
"""
import argparse
import json
import os

import pandas as pd
import torch

from classifier import Classifier
from neuron_search import NeuronMap, NeuronSearch
from perturber import Perturber
from tasks import TASKS


def get_args():
    ap = argparse.ArgumentParser()
    # data and task
    ap.add_argument("--task", default="asap2", choices=list(TASKS))
    ap.add_argument("--data_path", help="default: the task's own")
    ap.add_argument("--limit", type=int, help="only the first N samples")
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--include_source_text", action="store_true", help="asap2: add the source text(s) to the prompt")
    ap.add_argument("--source_texts_dir", default="data/asap2/source_texts")
    # models
    ap.add_argument("--model", required=True, help="classifier (HF name or local path)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--perturber_model", default="gpt-5-mini", help="model name (for vLLM: its --served-model-name)")
    ap.add_argument("--perturber_base_url", help="OpenAI-compatible server, e.g. vLLM: http://127.0.0.1:8000/v1 (default: OpenAI API)")
    # neuron map (extract_neurons/ output for the same model)
    ap.add_argument("--map_dir", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--a_star_n", type=int, default=10, help="a* = mean of the top N activations of the neuron")
    # neuron search
    ap.add_argument("--k", type=int, required=True, help="new neurons per iteration")
    ap.add_argument("--n_iter", type=int, required=True, help="max perturbation rounds per sample")
    ap.add_argument("--strategy", default="gradient", choices=["gradient", "negative", "random"],
                    help="negative and random are ablations")
    ap.add_argument("--use_negative", action="store_true", help="also lower neurons with negative gradients")
    ap.add_argument("--pooling", default="max", choices=["max", "mean", "topk"], help="pooling of S over the text tokens")
    ap.add_argument("--pool_topk", type=int, default=5, help="positions averaged by --pooling topk")
    ap.add_argument("--causal_candidates", type=int, default=50, help="candidates re-ranked by patching them to a*")
    ap.add_argument("--no_causal_check", action="store_true", help="rank by S only")
    ap.add_argument("--n_contexts", type=int, default=10)
    ap.add_argument("--context_before", type=int, default=120)
    ap.add_argument("--context_after", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def perturb(row, args, task, clf, search, neuron_map, perturber):
    """Run the iterative perturbation on one sample; None if it is already classified as the target."""
    text = row["text"]
    state = clf.run(task.messages(row, text), text)
    original = state.pred
    if original == args.target_label:
        return None
    used = {}  # key -> (Neuron, iteration it was chosen in)
    neuron_trace, score_trace = [], []
    for it in range(1, args.n_iter + 1):
        current = state.pred
        new = search.select(state, args.target_label, current, args.k, {(n.layer, n.index) for n, _ in used.values()})
        if not new:
            print(f"  [{row['id']}] no neuron left to use", flush=True)
            break
        used.update({n.key: (n, it) for n in new})
        contexts = [(n, neuron_map.contexts(n, args.n_contexts, args.context_before, args.context_after)) for n in new]
        text = perturber(text, task.description, current, args.target_label, contexts)
        new_state = clf.run(task.messages(row, text), text)

        prev_acts, now_acts = state.activations(), new_state.activations()
        step = {}
        for key, (n, since) in used.items():
            prev, now = search.pooled_activation(prev_acts, n), search.pooled_activation(now_acts, n)
            gap = n.a_star - prev
            step[key] = {"dir": n.direction, "since": since, "prev": prev, "now": now, "a_star": n.a_star,
                         "gap_closed": (now - prev) / gap if gap else None}
        neuron_trace.append(step)
        score_trace.append({"J_prev": clf.margin(state, args.target_label, current),
                            "J_now": clf.margin(new_state, args.target_label, current), "score_now": new_state.pred})
        print(f"  [{row['id']}] iteration {it}: {current} -> {new_state.pred}, neurons {[n.key for n in new]}", flush=True)
        state = new_state
        if state.pred == args.target_label:
            break

    return {"id": row["id"], "perturbed_text": text, "n_iterations": len(score_trace), "neurons": ";".join(used),
            "original_score": original, "final_score": state.pred,
            "neuron_trace": json.dumps(neuron_trace), "score_trace": json.dumps(score_trace)}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    task = TASKS[args.task](args)
    assert args.target_label in task.labels, f"--target_label must be one of {task.labels}"
    df = task.load(args.data_path or task.data_path)
    if args.limit:
        df = df.head(args.limit)
    done = set(pd.read_csv(args.output, usecols=["id"]).id.astype(str)) if os.path.exists(args.output) else set()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = Classifier(args.model, task.labels, getattr(torch, args.dtype), device)
    neuron_map = NeuronMap(args.map_dir, args.corpus_dir, clf.tok, clf.n_layers, clf.d_ff, args.a_star_n)
    search = NeuronSearch(clf, neuron_map, args.strategy, args.use_negative, args.pooling, args.pool_topk,
                          0 if args.no_causal_check else args.causal_candidates, args.seed)
    perturber = Perturber(args.perturber_model, args.perturber_base_url)

    for i, row in df.iterrows():
        if str(row["id"]) in done:
            continue
        result = perturb(row, args, task, clf, search, neuron_map, perturber)
        if result is None:
            print(f"[{i}] {row['id']}: already {args.target_label}, skipped", flush=True)
            continue
        print(f"[{i}] {row['id']}: {result['original_score']} -> {result['final_score']} in {result['n_iterations']} iterations", flush=True)
        pd.DataFrame([result]).to_csv(args.output, mode="a", header=not os.path.exists(args.output), index=False)


if __name__ == "__main__":
    main()
