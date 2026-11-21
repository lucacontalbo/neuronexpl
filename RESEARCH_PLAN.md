# Neuron-guided attacks on LLM-as-a-Judge: research plan

## Idea
A white-box neuron search selects MLP neurons of the judge (defender) whose activation change is predicted
(gradient x activation gap, S = g·(a* − a)) and verified (causal patching) to move the verdict toward a target
label. A perturber LLM (gpt-oss-20b) edits the input so that it reproduces the natural triggers of those neurons,
taken from a neuron map (top-activating contexts over a reference corpus).

## Setup

**Judges** (open-weight, instruction-tuned, dense; 3 families × 2 sizes)

| size | Qwen | Gemma | Mistral |
|---|---|---|---|
| small | Qwen3.5-0.8B | gemma-3-4b-it | Ministral-3-3B-Instruct-2512 |
| medium (14–40B) | Qwen3.5-27B | gemma-3-27b-it | Mistral-Small-3.2-24B-Instruct-2506 |

Large (>100B) models are left out: the available ones are MoE, where a neuron exists only on the tokens routed to
its expert, which the neuron search and the map do not support.

**Datasets** (LLM-as-a-Judge tasks)
- ASAP 2.0: pointwise essay scoring (1–6).
- BiGGen Bench (human_eval split): pointwise rubric-based grading of a response (1–5).
- JudgeBench (gpt + claude splits): pairwise comparison (A / B); response B is always the edited one.

**Strategies**: best (highest S, causal check), worst (lowest S), random neurons.

**Metrics**: attack success rate (ASR, prediction = target), ΔJ with J = logit(target) − logit(current),
paired comparison best vs worst vs random (Wilcoxon), trigger reproduction rate (selected neurons actually
activated by the edit), edit size.

## Part 1: Directionality
Does following the gradient direction of the selected neurons matter?
- Every judge × dataset × strategy, for every target class of each dataset.
- Expected: best > random > worst in ASR and ΔJ, beyond the generic effect of editing the text.

## Part 2: Specialised neuron maps
Build three neuron maps per judge, from different corpora:
- general (current corpus: web, multilingual, code, math);
- persuasion (PAP persuasion taxonomy, Persuasion-for-Good);
- bias (StereoSet, CrowS-Pairs).

Questions:
1. Do persuasion-specific neurons/strategies raise ASR over the general map?
2. Explaining the neurons selected from each map (and the strategies they induce in the edits), can we
   characterise the judge's weaknesses and biases?

## Part 3: Competitiveness
Compare with attacks on LLM judges from the literature (baselines: TBD):
- ASR;
- variability of the weaknesses and strategies found by each approach.
