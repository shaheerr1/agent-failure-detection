# Audit experiment results

Checkpoint: `final_model` | test traces: 87 | device: cuda

## 1. Baseline verification

Test set: 87 traces. Device: cuda.

| class            | precision | recall | f1    | support |
|------------------|-----------|--------|-------|---------|
| HALLUCINATION    | 0.667     | 0.727  | 0.696 | 22      |
| LOOP             | 0.824     | 0.875  | 0.848 | 16      |
| SUCCESS          | 0.783     | 0.692  | 0.735 | 26      |
| UNSAFE_EXECUTION | 1.0       | 1.0    | 1.0   | 23      |

**macro F1 = 0.82**  (95% bootstrap CI 0.734 to 0.891)  | accuracy = 0.816

Dissertation reported macro F1 0.820, accuracy 0.816.

Pipeline fidelity check: scoring the stored `text_truncated` column gives macro F1 0.82 and agrees with re-truncated raw text on 100.0% of traces.


Confusion matrix (rows true, cols predicted), order HALLUCINATION, LOOP, SUCCESS, UNSAFE_EXECUTION:
```
HALLUCINATION        16    1    5    0
LOOP                  2   14    0    0
SUCCESS               6    2   18    0
UNSAFE_EXECUTION      0    0    0   23
```

## 2. Shortcut ablations

Same checkpoint, same test set, input text transformed.

| variant              | macro F1 | delta  | accuracy | traces changed |
|----------------------|----------|--------|----------|----------------|
| baseline             | 0.82     | -      | 0.816    | 0              |
| V1 tool names masked | 0.82     | +0.000 | 0.816    | 36             |
| V2 mock markers gone | 0.82     | +0.000 | 0.816    | 25             |
| V3 FINAL: normalised | 0.784    | -0.036 | 0.793    | 60             |
| V4 all three         | 0.784    | -0.036 | 0.793    | 74             |

Per-class F1, delta against baseline in brackets:

| variant              | HALLUCINATION  | LOOP           | SUCCESS        | UNSAFE_EXECUTION |
|----------------------|----------------|----------------|----------------|------------------|
| baseline             | 0.696          | 0.848          | 0.735          | 1.000            |
| V1 tool names masked | 0.696 (+0.000) | 0.848 (+0.000) | 0.735 (+0.000) | 1.000 (+0.000)   |
| V2 mock markers gone | 0.696 (+0.000) | 0.848 (+0.000) | 0.735 (+0.000) | 1.000 (+0.000)   |
| V3 FINAL: normalised | 0.541 (-0.155) | 0.824 (-0.024) | 0.772 (+0.037) | 1.000 (+0.000)   |
| V4 all three         | 0.541 (-0.155) | 0.824 (-0.024) | 0.772 (+0.037) | 1.000 (+0.000)   |

**How to read this.** A large negative delta on UNSAFE_EXECUTION under V1 means the
model was reading tool names rather than behaviour. A large negative delta on LOOP
under V3 means it was reading the truncation artefact rather than repetition. Deltas
near zero mean the criticism does not hold for that class.

## 3. Runtime false positive rate

The gap in the original evaluation: SUCCESS traces were never fed to the
step-by-step detector, so the alarm rate on healthy runs was never measured.

| metric                               | value        |
|--------------------------------------|--------------|
| healthy traces swept                 | 26           |
| total prefixes evaluated             | 53           |
| prefixes predicting a failure        | 7 (13.2%)    |
| healthy traces alarmed at least once | 5/26 (19.2%) |
| alarms per 100 healthy traces        | 19           |

Which class the false alarms claim:

| falsely predicted class | prefix count |
|-------------------------|--------------|
| HALLUCINATION           | 6            |
| LOOP                    | 1            |

Runtime alarm precision across the whole test set: **0.942** (114 correct failure predictions against 7 false alarms, counted per prefix).


**How to read this.** If healthy traces alarm often, the 78% and 63% early-detection
figures are not evidence of a usable monitor, because the detector is simply biased
toward failure classes on partial input. A low false positive rate here is the single
strongest result the project can produce.
