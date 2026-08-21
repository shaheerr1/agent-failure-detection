# Monitoring Dashboard

A static viewer over the project's frozen evaluation artefacts, built as a
single-page React application. It corresponds to Section 5.5 of the dissertation.

The dashboard **runs no inference**. Every figure it displays is read from
`dashboard_data.js`, which is generated once, offline, by
`export_dashboard_data.py`. This keeps the demonstrable artefact isolated from
the serving infrastructure: the classifier is a local PyTorch model requiring a
GPU, and embedding it in a web application was outside the scope of the
contribution.

## Running it

Open `index.html` in a browser. That is the whole procedure — no build step, no
`npm install`, no local server. React is vendored in `vendor/` so the page also
works with no network connection, which matters when demonstrating it live.

## Regenerating the data

From the project root, with the venv active:

```bash
python demo/export_dashboard_data.py
```

This reads:

| Artefact | Supplies |
|---|---|
| `data_splits/dataset_clean_433.xlsx` | class distribution, provenance |
| `data_splits/test.xlsx`              | trace content for the replay panel |
| `experiments/runtime_results.xlsx`   | per-trace detection steps |
| `classifier/final_model/`            | per-step class probabilities |

and writes `demo/dashboard_data.js`.

If `torch`/`transformers` or the trained model are unavailable, the script still
runs and produces everything except the per-step probabilities; the replay panel
then shows the trace stepping and the detection marker, and says so plainly.

A `.js` file is emitted rather than `.json` because browsers block `fetch()` of
local `.json` under the `file://` protocol, whereas a `<script src>` tag loads
without a server.

## The three panels

**Overview** — the taxonomy with per-class definitions and counts, the cleaned
class distribution, and the position of the model in the five-model baseline
ladder.

**Trace replay** — the panel that justifies the dashboard's existence. The
classifier's output is shown at each prefix of a trace, with the final answer
withheld exactly as during genuine runtime. Stepping through a loop trace shows
the loop probability sitting near zero while the searches differ, then jumping
at the step where the agent re-fetches a page it has already seen. This makes
the runtime-detection argument observable rather than merely tabulated.

**Results** — per-class precision/recall/F1 with support, the confusion matrix,
the full five-model comparison, and runtime detection rates by failure type.

## Files

```
demo/
  index.html                 markup and design tokens
  app.js                     the application (React.createElement, no JSX)
  dashboard_data.js          generated — do not edit by hand
  export_dashboard_data.py   generator
  vendor/                    React 18.3.1 UMD builds, vendored for offline use
```

## Notes on the figures

The runtime panel reports two different quantities, and they are not
interchangeable. *Detected before final step* is the runtime measure. *Average
detection point* is conditional on detection having occurred at all, so for
hallucination it describes only the minority of traces that were caught, and
must be read alongside the rate rather than independently of it.
