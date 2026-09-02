# Runtime Detection of Failure Modes in LLM Agents

Detecting agent failures from execution traces **while the agent is still running**, rather than after the run completes.

A DeBERTa-v3 classifier is fine tuned on 433 hand labelled ReAct agent traces across four classes, then evaluated the way it would actually be deployed: on partial traces, one step at a time, with the final answer withheld.

| | |
|---|---|
| Test macro F1 | **0.820** (87 trace held out set, no group leakage) |
| Best baseline | RoBERTa-base 0.756, TF-IDF + LR 0.664, frozen MiniLM + LR 0.464, majority 0.115 |
| Caught before the agent finished | 78% of unsafe actions, 63% of loops, 32% of hallucinations |

MSc Applied AI research, London South Bank University. Supervisor: Dr. Ali Salimian.

---

## Why runtime and not post hoc

Classifying a finished run is easy once the final answer exists, and useless for prevention. By the time a completed trace reaches a dashboard, the email has been sent and the purchase has been made.

The question this repo answers is narrower and harder: is the failure already legible in the trace before the agent gets there, and how early.

---

## Failure taxonomy

Four classes, applied to whole traces. Definitions are enforced by the annotation guide in `annotation/`.

| Class | Signal in the trace |
|---|---|
| `SUCCESS` | Final answer is grounded in the observations the tools actually returned |
| `HALLUCINATION` | Specific claims that appear in no observation |
| `LOOP` | Same call repeated with no state change and no progress |
| `UNSAFE_EXECUTION` | Action taken without authority or on fabricated parameters (mock purchase, mock delete, mock send) |

Unsafe actions are generated safely: the agent's action tools are mocks that log the requested action instead of performing it.

---

## Results

### Model comparison, identical frozen test split

| Class | Majority | MiniLM + LR | TF-IDF + LR | RoBERTa | **DeBERTa-v3** |
|---|---|---|---|---|---|
| HALLUCINATION | 0.000 | 0.586 | 0.678 | 0.703 | 0.696 |
| LOOP | 0.000 | 0.353 | 0.476 | **0.970** | 0.848 |
| SUCCESS | 0.460 | 0.359 | 0.619 | 0.667 | **0.735** |
| UNSAFE_EXECUTION | 0.000 | 0.558 | 0.885 | 0.686 | **1.000** |
| **Macro F1** | 0.115 | 0.464 | 0.664 | 0.756 | **0.820** |

Two findings worth more than the ranking:

**Bag of words cannot represent a loop.** TF-IDF reaches 0.885 on unsafe execution but 0.476 on loops, catching 5 of 16. A loop is a repetition pattern, not a token pattern, and a model that discards order cannot express "this happened three times". That is the empirical argument for a contextual encoder on this task.

**DeBERTa's edge over RoBERTa is not uniform.** It sits almost entirely in `UNSAFE_EXECUTION` (1.000 vs 0.686). RoBERTa is the better loop detector (0.970 vs 0.848). No model exceeds roughly 0.70 on hallucination.

### Runtime detection, 61 failure traces from the held out set

Each trace is replayed prefix by prefix. The final answer line is stripped from every partial input, since at runtime it does not exist yet. The detection step is the first prefix at which the prediction matches the true label.

| Failure type | Detected at any step | Detected before final step | Avg detection point |
|---|---|---|---|
| `UNSAFE_EXECUTION` | 23 / 23 | 18 / 23 (78%) | 59% through the trace |
| `LOOP` | 14 / 16 | 10 / 16 (63%) | 72% through the trace |
| `HALLUCINATION` | 11 / 22 | 7 / 22 (32%) | 56% through the trace |

Stepping through the per prefix probability distributions shows three different detection mechanisms rather than one capability applied three times:

* **Unsafe execution** fires on the precursor setup, before the unsafe call happens. This is anticipatory, and it is shortcut prone. Reported as precursor detection, not act detection.
* **Loop** fires on evidence, once a repeated observation is actually visible. This is the cleanest and most defensible runtime result.
* **Hallucination** resolves late, often after passing through a loop like phase, because there is nothing to detect until the ungrounded answer materialises.

Runtime detection is not one problem. It is three, with different ceilings.

---

## Dataset integrity

Two defects were found before training. Both inflate the headline number, and neither announces itself.

**Synthetic template collapse.** Augmentation of the two minority classes had collapsed into near duplicate families. One lineage contributed 40 rows following an identical template, with only the product name varying. Forty copies of one signal is not forty examples, and an early split had put the whole family in the test set. Fix: embed every trace with `all-MiniLM-L6-v2` and keep the 12 most mutually dissimilar members per lineage by greedy farthest point sampling. 592 rows down to 433.

**Broken lineage grouping.** `Parent Trace ID` was stored as comma separated strings, so grouping on the raw value produced overlapping pseudo groups and let sibling traces land on opposite sides of the train and test boundary. The leakage check passed vacuously because it compared group strings rather than actual parents. Fix: rebuild groups as connected components over shared parents, then split with `StratifiedGroupKFold` on the true group.

**Length.** 23.6% of traces exceeded the 512 token limit. Head and tail truncation is used, always preserving the task line and the final answer.

Post cleaning class counts: SUCCESS 129, UNSAFE_EXECUTION 115, HALLUCINATION 109, LOOP 80 (largest to smallest ratio 1.6:1). Splits: train 259 / val 87 / test 87, verified for zero group overlap and near identical per class proportions.

---

## Repo layout

```
agent/          ReAct agent, tool definitions, task lists per failure category
tools/          Real tools plus mock action tools used to generate unsafe traces safely
annotation/     Review scripts, annotation guide, agreement analysis
data/           Raw and labelled traces (JSON, gitignored by default)
data_splits/    Frozen train / val / test splits, baseline predictions, model comparison
classifier/     Fine tuning and evaluation for DeBERTa-v3 and RoBERTa
experiments/    Runtime replay, per step probability inspection, cross model trace generation
notebooks/      EDA and results figures
demo/           Monitoring dashboard
writeup/        Dissertation chapters and figures
```

Model weights are not tracked. `classifier/checkpoints*/` and `final_model*/` are gitignored, so the classifier must be retrained locally to reproduce inference.

---

## Reproducing

Environment: Python 3.12, PyTorch 2.6.0 with CUDA. `random_state=42` throughout.

```bash
pip install -r requirements.txt

# 1. generate traces (needs an agent API key in .env)
python run_pipeline.py

# 2. review and label
python annotation/review_traces.py

# 3. train and evaluate
python classifier/train.py
python classifier/evaluate.py

# 4. runtime experiment
python experiments/test_classifier_runtime.py     # aggregate, writes runtime_results.xlsx
python experiments/inspect_runtime.py             # per step probabilities, one trace per class
```

Steps 3 and 4 run against the frozen splits in `data_splits/`, so the reported numbers are reproducible without regenerating data.

---

## Limitations

* `UNSAFE_EXECUTION` and `LOOP` remain partly template derived. Diversity sampling reduces redundancy, it does not create diversity that was never generated.
* The unsafe detector fires on the precursor pattern, which means it is exposed to false positives on runs that set up the same way and then behave correctly.
* The classifier is trained on complete traces and applied to partial ones. That it transfers at all is a positive result, but training on prefixes directly should reduce detection latency.
* Detection point is measured in agent steps, not seconds. It reports earliness in the reasoning sequence, not wall clock speed.
* Detection is credited at the first correct prediction, without requiring the prediction to stay stable afterwards. A stricter definition would report later detection.
* Labels are trace level, so a step level ground truth would be a stronger basis for runtime claims.
* Hallucination sits at roughly 0.70 for every model tried. The residual errors are high confidence faithfulness failures, which single sequence classification is structurally unable to see. The next step is a claim versus evidence check, not a bigger classifier.

---

## What is reusable here

The taxonomy and annotation guide, the lineage aware splitting procedure, and the runtime replay protocol are the parts that transfer to other agent stacks. The trained weights are specific to this agent and tool set.
