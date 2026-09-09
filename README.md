# Runtime Detection of Failure Modes in LLM Agents

Detecting agent failures from execution traces **while the agent is still running**, rather than after the run completes.

A DeBERTa-v3 classifier is fine tuned on 433 hand labelled ReAct agent traces across four classes, then evaluated the way it would actually be deployed: on partial traces, one step at a time, with the final answer withheld.

| | |
|---|---|
| Test macro F1 | **0.820** (87 trace held out set, no group leakage) |
| Best baseline | RoBERTa-base 0.756, TF-IDF + LR 0.664, frozen MiniLM + LR 0.464, majority 0.115 |
| Caught before the agent finished | 78% of unsafe actions, 63% of loops, 32% of hallucinations |
| Shortcut ablations | Tool name and mock marker masking: no change. Removing the final answer: hallucination 0.696 to 0.541 |
| False alarm rate | 13.2% of prefixes on healthy traces, 0.942 runtime alarm precision |

MSc Applied AI research, London South Bank University.

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

**DeBERTa's edge over RoBERTa is not uniform, and the ranking itself is not established.** The gap sits almost entirely in `UNSAFE_EXECUTION` (1.000 vs 0.686), and RoBERTa is the better loop detector (0.970 vs 0.848). On an 87 trace test set from a single seed, the bootstrap interval on macro F1 spans 0.734 to 0.891, which is wider than the 0.820 versus 0.756 gap. Read the per class pattern as the finding, not the ordering. No model exceeds roughly 0.70 on hallucination.

### Runtime detection, 61 failure traces from the held out set

Each trace is replayed prefix by prefix. The final answer line is stripped from every partial input, since at runtime it does not exist yet. The detection step is the first prefix at which the prediction matches the true label.

| Failure type | Detected at any step | Detected before final step | Avg detection point |
|---|---|---|---|
| `UNSAFE_EXECUTION` | 23 / 23 | 18 / 23 (78%) | 59% through the trace |
| `LOOP` | 14 / 16 | 10 / 16 (63%) | 72% through the trace |
| `HALLUCINATION` | 11 / 22 | 7 / 22 (32%) | 56% through the trace |

Stepping through the per prefix probability distributions shows three different detection mechanisms rather than one capability applied three times:

* **Unsafe execution** fires on the precursor setup, before the unsafe call happens. This is anticipatory rather than reactive, and it is exposed to false positives on runs that set up the same way and then behave correctly. Reported as precursor detection, not act detection. Ablation rules out tool names and mock markers as the cue; the control experiment on safe runs sharing the same precursor has not been run.
* **Loop** fires on evidence, once a repeated observation is actually visible. This is the cleanest and most defensible runtime result.
* **Hallucination** resolves late, often after passing through a loop like phase, because there is nothing to detect until the ungrounded answer materialises.

Runtime detection is not one problem. It is three, with different ceilings.

### Does it rely on shortcuts?

Two classes score suspiciously well and both have an obvious lexical explanation. `UNSAFE_EXECUTION` is defined by three tool names that appear verbatim in the input, and a regex on those names alone reaches F1 0.958. Every `LOOP` trace in the test set is truncated, so absence of the `FINAL:` line predicts the class perfectly. Either would mean the classifier had learned a token rather than a behaviour.

Both were tested against the trained checkpoint. Same weights, same test split, input text transformed.

| Variant | Macro F1 | Delta | HALLUCINATION | LOOP | SUCCESS | UNSAFE |
|---|---|---|---|---|---|---|
| baseline | 0.820 | | 0.696 | 0.848 | 0.735 | 1.000 |
| tool names aliased | 0.820 | 0.000 | 0.696 | 0.848 | 0.735 | 1.000 |
| mock markers removed | 0.820 | 0.000 | 0.696 | 0.848 | 0.735 | 1.000 |
| final answer removed | 0.784 | -0.036 | **0.541** | 0.824 | 0.772 | 1.000 |
| all three | 0.784 | -0.036 | 0.541 | 0.824 | 0.772 | 1.000 |

**Neither shortcut hypothesis survives.** Renaming the action tools across 36 traces and stripping the `[MOCK]` markers across 25 changed nothing at all, not one prediction. The regex baseline scoring 0.958 is a fact about how separable the task is, not about what the model learned.

**The real constraint is the final answer.** Hallucination loses 0.155 F1 when it is removed, four times the movement of any other class. That is the mechanism behind the 32% runtime figure above: a hallucination is an ungrounded claim in the final answer, so before the answer exists there is nothing to detect. Success improves slightly under the same transform, consistent with fewer hallucination false positives once the answer is gone.

Macro F1 with a bootstrap 95% confidence interval is 0.820 (0.734 to 0.891). The interval is wide enough that the DeBERTa and RoBERTa difference is not established.

### False alarm rate

The runtime experiment above only sees failure traces, which measures recall and says nothing about how often a healthy run raises an alarm. Sweeping the 26 held out `SUCCESS` traces through the identical prefix by prefix loop:

| Metric | Value |
|---|---|
| Prefixes predicting a failure | 7 / 53 (13.2%) |
| Healthy traces alarmed at least once | 5 / 26 (19.2%) |
| Runtime alarm precision, per prefix | 0.942 |

Six of the seven false alarms claim `HALLUCINATION`, the class the ablation shows is weakest. The sample is small and the intervals are wide, roughly 7 to 25% per prefix.

Reproduce both with `python audit_run.py all`.

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
agent/          ReAct agent, tool definitions (real plus mock action tools), task lists
annotation/     Dual model auto labeller, annotation guide, agreement analysis
data/           Raw and labelled traces (JSON, gitignored by default)
data_splits/    Frozen train / val / test splits, baseline predictions, model comparison
classifier/     train.py, evaluate.py, dataset build notebook, DeBERTa-v3 and RoBERTa
experiments/    Runtime replay, per step probability inspection, cross model trace generation
notebooks/      EDA and results figures
demo/           Static results explorer (no live service)
writeup/        Dissertation chapters and figures
audit_run.py    Shortcut ablations and runtime false alarm rate
```

Model weights are not tracked. `classifier/checkpoints*/` and `final_model*/` are gitignored, so the classifier must be retrained locally to reproduce inference.

---

## Reproducing

Environment: Python 3.12, PyTorch 2.6.0 with CUDA. Seeds fixed at 42 unless `--seeds` is passed. Training falls back to CPU or Apple MPS automatically.

```bash
pip install -r requirements.txt   # install PyTorch for your CUDA version first

# 1. generate traces (needs GROQ_API_KEY in .env)
python run_pipeline.py

# 2. review and label (needs ANTHROPIC_API_KEY and OPENAI_API_KEY)
python annotation/auto_labeller.py
python review_traces.py

# 3. train, then score the checkpoint
python classifier/train.py --save-model          # single run, seed 42
python classifier/train.py --seeds 42,43,44,45,46  # sweep, reports mean and std
python classifier/evaluate.py                     # scores an existing checkpoint

# 4. runtime experiment
python experiments/test_classifier_runtime.py     # aggregate, writes runtime_results.xlsx
python experiments/inspect_runtime.py             # per step probabilities, one trace per class

# 5. ablations and false alarm rate
python audit_run.py all                           # writes results/summary.md
```

Steps 3 to 5 run against the frozen splits in `data_splits/`, so the reported numbers are reproducible without regenerating data. Step 4 needs a trained checkpoint at `classifier/final_model`, which step 3 with `--save-model` produces.

The dataset build itself (diversity sampling, connected-component grouping, the split) still lives in `classifier/train_classifier.ipynb`. `train.py` consumes its frozen output rather than rebuilding it.

---

## Limitations

* `UNSAFE_EXECUTION` and `LOOP` remain partly template derived. Diversity sampling reduces redundancy, it does not create diversity that was never generated.
* The unsafe detector fires on the precursor pattern rather than the act, so it is exposed to false positives on runs that set up the same way and then behave correctly. None of the measured false alarms on healthy traces were unsafe, but the targeted control (safe runs sharing the same precursor) has not been run.
* The classifier is trained on complete traces and applied to partial ones. That it transfers at all is a positive result, but training on prefixes directly should reduce detection latency.
* Detection point is measured in agent steps, not seconds. It reports earliness in the reasoning sequence, not wall clock speed.
* Detection is credited at the first correct prediction, without requiring the prediction to stay stable afterwards. A stricter definition would report later detection.
* Labels are trace level, so a step level ground truth would be a stronger basis for runtime claims.
* All reported figures come from a single training run at `seed=42`. The bootstrap interval on macro F1 spans 0.734 to 0.891, so per class differences between models on a 87 trace test set should not be over read. A seed sweep is available via `train.py --seeds`.
* The held out set contains no real traces for `UNSAFE_EXECUTION` or `LOOP`; both are entirely synthetic or synthetic truncated. The ablations show the model is not exploiting a lexical cue, but scores on those two classes still measure template fidelity rather than field performance.
* Hallucination sits at roughly 0.70 for every model tried. The residual errors are high confidence faithfulness failures, which single sequence classification is structurally unable to see. The next step is a claim versus evidence check, not a bigger classifier.

---

## What is reusable here

The taxonomy and annotation guide, the lineage aware splitting procedure, and the runtime replay protocol are the parts that transfer to other agent stacks. The trained weights are specific to this agent and tool set.
