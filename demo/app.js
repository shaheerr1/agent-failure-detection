/* ===========================================================================
 * Agent Failure Detection — Monitoring Dashboard
 * ---------------------------------------------------------------------------
 * A static viewer over the project's frozen evaluation artefacts. It runs no
 * inference: every number shown is read from dashboard_data.js, which is
 * produced by demo/export_dashboard_data.py.
 *
 * Written against React.createElement directly (no JSX, no build step) so the
 * page opens straight from the filesystem with no toolchain.
 * ======================================================================== */
(function () {
  "use strict";

  var h = React.createElement;
  var useState = React.useState, useMemo = React.useMemo,
      useEffect = React.useEffect, useRef = React.useRef;

  var D = window.DASHBOARD_DATA;

  /* ── class identity ────────────────────────────────────────────────────── */
  var CLASS_META = {
    SUCCESS:          { label: "Success",          v: "--c-success",
      blurb: "Every substantive claim in the final answer is supported by a tool observation, or the agent honestly reports that the information could not be found." },
    HALLUCINATION:    { label: "Hallucination",    v: "--c-halluc",
      blurb: "The final answer asserts a specific fact appearing in no observation, irrespective of whether that fact happens to be true in the world." },
    LOOP:             { label: "Loop",             v: "--c-loop",
      blurb: "The agent repeats an action three or more times — identical queries or semantic rephrasings — without advancing towards the task." },
    UNSAFE_EXECUTION: { label: "Unsafe execution", v: "--c-unsafe",
      blurb: "The agent invokes a consequential mock action tool without explicit authorisation for that action in the task description." }
  };
  var ORDER = ["SUCCESS", "HALLUCINATION", "LOOP", "UNSAFE_EXECUTION"];

  function cvar(name) { return "var(" + name + ")"; }
  function colorOf(cls) {
    var m = CLASS_META[cls];
    return m ? cvar(m.v) : "var(--text-3)";
  }
  function labelOf(cls) {
    var m = CLASS_META[cls];
    return m ? m.label : cls;
  }
  function pct(x, dp) { return (x * 100).toFixed(dp == null ? 0 : dp) + "%"; }
  function f3(x) { return x == null ? "—" : x.toFixed(3); }

  /* ── hover layer ───────────────────────────────────────────────────────────
     An HTML chart is interactive by default: every mark carries a tooltip.
     One shared floating element, positioned from pointer coordinates, rather
     than a per-mark title attribute (which is slow to appear and unstyleable). */
  function useTooltip() {
    var [tip, setTip] = useState(null);
    var show = function (e, content) {
      setTip({ x: e.clientX, y: e.clientY, content: content });
    };
    var hide = function () { setTip(null); };
    var node = tip ? h("div", {
      style: {
        position: "fixed", left: 0, top: 0,
        transform: "translate(" + (tip.x + 14) + "px," + (tip.y - 12) + "px)",
        background: "var(--surface-1)", color: "var(--text-1)",
        border: "1px solid var(--border-str)", borderRadius: "6px",
        padding: "7px 10px", fontSize: "12.5px", lineHeight: 1.45,
        boxShadow: "0 6px 20px rgba(0,0,0,.18)", pointerEvents: "none",
        zIndex: 999, maxWidth: "260px", whiteSpace: "nowrap"
      }
    }, tip.content) : null;
    return { show: show, hide: hide, node: node };
  }

  /* ── small pieces ──────────────────────────────────────────────────────── */
  function Swatch(p) {
    return h("span", { className: "swatch", style: { background: p.color } });
  }

  function Tile(p) {
    return h("div", { className: "tile" },
      h("div", { className: "k" }, p.k),
      h("div", { className: "v", style: p.color ? { color: p.color } : null }, p.v),
      p.n ? h("div", { className: "n" }, p.n) : null);
  }

  function Legend(p) {
    return h("div", { className: "legend" },
      p.items.map(function (it) {
        return h("span", { key: it.label, style: { display: "flex", alignItems: "center" } },
          h(Swatch, { color: it.color }), it.label);
      }));
  }

  /* Horizontal bars. Every bar is directly labelled with its value, which is
     also what discharges the contrast relief rule for the lighter hues. */
  function BarChart(p) {
    var max = p.max != null ? p.max : Math.max.apply(null, p.rows.map(function (r) { return r.value; }));
    var fmt = p.format || function (v) { return String(v); };
    var labelW = p.labelWidth || 132;
    var tt = useTooltip();
    return h("div", null,
      tt.node,
      p.rows.map(function (r, i) {
        var w = max > 0 ? (r.value / max) * 100 : 0;
        return h("div", {
          key: r.name + i,
          onMouseMove: function (e) {
            tt.show(e, h("span", null,
              h("strong", null, r.name), " — ", fmt(r.value),
              r.note ? h("span", { style: { color: "var(--text-3)" } }, "  ·  " + r.note) : null));
          },
          onMouseLeave: tt.hide,
          style: { display: "grid", gridTemplateColumns: labelW + "px 1fr auto",
                   alignItems: "center", gap: "12px", marginBottom: "9px",
                   cursor: "default", borderRadius: "5px" }
        },
          h("div", { style: { fontSize: "12.5px", color: "var(--text-2)",
                              whiteSpace: "nowrap", overflow: "hidden",
                              textOverflow: "ellipsis" } },
            r.color ? h(Swatch, { color: r.color }) : null, r.name),
          h("div", { style: { height: "17px", background: "var(--surface-2)",
                              borderRadius: "4px", overflow: "hidden" } },
            h("div", { style: { width: w + "%", height: "100%",
                                background: r.color || "var(--accent)",
                                borderRadius: "4px" } })),
          h("div", { className: "mono", style: {
              minWidth: "48px", textAlign: "right",
              fontWeight: r.emphasis ? 650 : 400,
              color: r.emphasis ? "var(--text-1)" : "var(--text-2)" } },
            fmt(r.value)));
      }));
  }

  /* ── panel 1: overview ─────────────────────────────────────────────────── */
  function Overview(p) {
    var ds = D.dataset, off = D.offline, bl = D.baselines;
    var byName = {};
    ds.classes.forEach(function (c) { byName[c.name] = c.count; });

    return h("div", null,
      h("section", { className: "block" },
        h("h2", null, "At a glance"),
        h("div", { className: "grid g4" },
          h(Tile, { k: "Test macro F1", v: off.macroF1.toFixed(3),
                    n: "held-out set, " + ds.splits.test + " traces" }),
          h(Tile, { k: "Accuracy", v: off.accuracy.toFixed(3), n: "same split" }),
          h(Tile, { k: "Dataset", v: String(ds.total),
                    n: ds.splits.train + " train / " + ds.splits.val + " val / " + ds.splits.test + " test" }),
          h(Tile, { k: "Failure classes", v: "4", n: "reduced from six, empirically" }))),

      h("section", { className: "block" },
        h("h2", null, "The taxonomy"),
        h("p", { className: "lede" },
          "Four classes survived into training. Categories are defined by observable symptoms in the trace rather than by internal cause, because a runtime detector sees only the thought–action–observation sequence emitted so far."),
        h("div", { className: "grid g2" },
          ORDER.map(function (cls) {
            var m = CLASS_META[cls];
            return h("div", { key: cls, className: "tax",
                              style: { "--bar": cvar(m.v) } },
              h("h3", null, m.label),
              h("p", null, m.blurb),
              h("div", { className: "cnt" },
                (byName[cls] || 0) + " traces · " + pct((byName[cls] || 0) / ds.total, 1) + " of the dataset"));
          }))),

      h("section", { className: "block" },
        h("h2", null, "Class distribution"),
        h("p", { className: "lede" },
          "After lineage down-sampling, the largest class is roughly 1.6 times the smallest. Class-weighted cross-entropy handles the residual imbalance at training time."),
        h("div", { className: "card" },
          h(BarChart, {
            rows: ORDER.map(function (cls) {
              return { name: labelOf(cls), value: byName[cls] || 0, color: colorOf(cls) };
            })
          }),
          ds.sources && ds.sources.length
            ? h("div", { className: "note" },
                "Provenance: " + ds.sources.map(function (s) {
                  return s.count + " " + s.name.toLowerCase().replace(/_/g, " ");
                }).join(" · "))
            : null)),

      h("section", { className: "block" },
        h("h2", null, "Where this model sits"),
        h("p", { className: "lede" },
          "Macro F1 on the identical held-out test set. The majority-class floor establishes that the task is not solvable by guessing; each rung above it removes one alternative explanation for the result."),
        h("div", { className: "card" },
          h(BarChart, {
            max: 1,
            labelWidth: 128,
            format: function (v) { return v.toFixed(3); },
            rows: bl.models.map(function (m) {
              return { name: m.short, value: m.macroF1,
                       color: m.kind === "model" ? "var(--accent)" : "var(--border-str)",
                       emphasis: m.kind === "model" };
            })
          }),
          h("div", { className: "note" },
            "The chosen model is highlighted. Full per-class breakdown is in the Results panel."))));
  }

  /* ── panel 2: trace replay ─────────────────────────────────────────────── */
  function ProbPanel(p) {
    if (!p.step || !p.step.probs) {
      return h("div", { className: "banner" },
        h("strong", null, "Per-step probabilities not exported. "),
        "Run ", h("span", { className: "mono" }, "python demo/export_dashboard_data.py"),
        " with the project venv active to populate this panel.");
    }
    var probs = p.step.probs;
    var sorted = ORDER.slice().sort(function (a, b) { return (probs[b] || 0) - (probs[a] || 0); });
    var top = sorted[0];
    return h("div", null,
      p.tt ? p.tt.node : null,
      sorted.map(function (cls, i) {
        var v = probs[cls] || 0;
        return h("div", {
          key: cls, className: "probrow" + (i === 0 ? " top" : ""),
          onMouseMove: p.tt ? function (e) {
            p.tt.show(e, h("span", null,
              h("strong", null, labelOf(cls)), " — ", v.toFixed(3),
              h("span", { style: { color: "var(--text-3)" } },
                "  ·  " + Math.round(v * 100) + "% of the mass")));
          } : null,
          onMouseLeave: p.tt ? p.tt.hide : null
        },
          h("div", { className: "nm" }, h(Swatch, { color: colorOf(cls) }), labelOf(cls)),
          h("div", { className: "track" },
            h("div", { className: "fill",
                       style: { width: (v * 100) + "%", background: colorOf(cls) } })),
          h("div", { className: "val" }, v.toFixed(3)));
      }),
      h("div", { style: { marginTop: "14px" } },
        p.trueLabel === top
          ? h("div", { className: "banner hit" },
              h("strong", null, "Correct at this step. "),
              "The classifier names ", labelOf(top),
              " on the partial trace, before the agent has answered.")
          : h("div", { className: "banner" },
              "Predicting ", h("strong", null, labelOf(top)),
              ". True label is ", h("strong", null, labelOf(p.trueLabel)), ".")));
  }

  function StepCard(p) {
    var s = p.step;
    var rows = [];
    if (s.thought) rows.push(["Thought", s.thought, false]);
    if (s.action)  rows.push(["Action", s.action + (s.input ? "  " + s.input : ""), false]);
    if (s.obs)     rows.push(["Observation", s.obs, true]);
    if (!rows.length) rows.push(["Step", "(no content parsed)", true]);
    return h("div", {
      className: "trace-step" + (p.state === "cur" ? " cur" : p.state === "future" ? " dim" : ""),
      ref: p.innerRef
    },
      h("div", { style: { display: "flex", justifyContent: "space-between",
                          alignItems: "baseline", marginBottom: "8px" } },
        h("span", { className: "pill" }, "Step " + s.n),
        p.detected ? h("span", { className: "pill",
            style: { borderColor: "var(--c-unsafe)", color: "var(--c-unsafe)" } },
          "first detected here") : null),
      rows.map(function (r, i) {
        return h("div", { key: i, className: "row" },
          h("div", { className: "lbl" }, r[0]),
          h("div", { className: "body" + (r[2] ? " obs" : "") }, r[1]));
      }));
  }

  function Replay(p) {
    var traces = D.replay || [];
    var [idx, setIdx] = useState(0);
    var [step, setStep] = useState(1);
    var [playing, setPlaying] = useState(false);
    var timer = useRef(null);
    var curRef = useRef(null);
    var probTip = useTooltip();

    var tr = traces[idx];
    var nSteps = tr ? tr.steps.length : 0;

    useEffect(function () { setStep(1); setPlaying(false); }, [idx]);

    useEffect(function () {
      if (!playing) return;
      timer.current = setTimeout(function () {
        setStep(function (s) {
          if (s >= nSteps) { setPlaying(false); return s; }
          return s + 1;
        });
      }, 1100);
      return function () { clearTimeout(timer.current); };
    }, [playing, step, nSteps]);

    useEffect(function () {
      if (curRef.current && curRef.current.scrollIntoView) {
        curRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }, [step]);

    if (!tr) {
      return h("div", { className: "block" },
        h("div", { className: "banner" }, "No replay traces in the data file."));
    }

    var cur = tr.steps[step - 1];

    var grouped = {};
    traces.forEach(function (t, i) {
      (grouped[t.label] = grouped[t.label] || []).push({ t: t, i: i });
    });

    return h("div", null,
      h("section", { className: "block" },
        h("h2", null, "Trace replay"),
        h("p", { className: "lede" },
          "The classifier is run once per prefix of the trace, with the final answer withheld — exactly as it would be during execution. Step through to watch the probability distribution move, and see the step at which the failure first becomes nameable."),

        h("div", { className: "controls" },
          h("select", {
            className: "sel", value: idx,
            onChange: function (e) { setIdx(Number(e.target.value)); }
          }, ORDER.filter(function (c) { return grouped[c]; }).map(function (cls) {
            return h("optgroup", { key: cls, label: labelOf(cls) },
              grouped[cls].map(function (o) {
                return h("option", { key: o.i, value: o.i },
                  o.t.traceId + " · " + o.t.steps.length + " steps");
              }));
          })),
          h("button", { className: "btn", disabled: step <= 1,
            onClick: function () { setPlaying(false); setStep(step - 1); } }, "‹ Back"),
          h("button", { className: "btn", disabled: step >= nSteps,
            onClick: function () { setPlaying(false); setStep(step + 1); } }, "Next ›"),
          h("button", {
            className: "btn pri",
            onClick: function () {
              if (playing) { setPlaying(false); return; }
              if (step >= nSteps) setStep(1);
              setPlaying(true);
            }
          }, playing ? "Pause" : "Play"),
          h("span", { style: { marginLeft: "4px", fontSize: "12.5px", color: "var(--text-3)" } },
            "Step ", h("strong", { style: { color: "var(--text-1)" } }, step),
            " of ", nSteps)),

        h("div", { className: "card", style: { marginBottom: "16px" } },
          h("div", { className: "lbl", style: {
              fontSize: "10.5px", fontWeight: 700, letterSpacing: ".08em",
              textTransform: "uppercase", color: "var(--text-3)", marginBottom: "4px" } },
            "Task"),
          h("div", { style: { fontSize: "14px" } }, tr.task),
          h("div", { style: { marginTop: "12px", display: "flex", gap: "8px", flexWrap: "wrap" } },
            h("span", { className: "pill",
              style: { borderColor: colorOf(tr.label), color: colorOf(tr.label) } },
              "true label · " + labelOf(tr.label)),
            h("span", { className: "pill" }, tr.source.toLowerCase().replace(/_/g, " ")),
            tr.detectedAt
              ? h("span", { className: "pill" }, "runtime experiment: first correct at step " + tr.detectedAt)
              : h("span", { className: "pill" }, "not detected before final step")),
          h("div", { className: "stepper" },
            tr.steps.map(function (s) {
              return h("button", {
                key: s.n, "aria-current": s.n === step,
                className: tr.detectedAt === s.n ? "detected" : "",
                onClick: function () { setPlaying(false); setStep(s.n); }
              }, s.n);
            }))),

        h("div", { className: "replay-grid" },
          h("div", { className: "scrollbox" },
            tr.steps.slice(0, step).map(function (s) {
              var isCur = s.n === step;
              return h(StepCard, {
                key: s.n, step: s, state: isCur ? "cur" : "past",
                detected: tr.detectedAt === s.n,
                innerRef: isCur ? curRef : null
              });
            }),
            step >= nSteps && tr.finalAnswer
              ? h("div", { className: "trace-step", style: { borderStyle: "dashed" } },
                  h("div", { className: "lbl" }, "Final answer (withheld from the classifier)"),
                  h("div", { className: "body" }, tr.finalAnswer))
              : null),

          h("div", null,
            h("div", { className: "card" },
              h("div", { style: { fontSize: "12px", textTransform: "uppercase",
                                  letterSpacing: ".06em", color: "var(--text-3)",
                                  fontWeight: 600, marginBottom: "12px" } },
                "Classifier output after step " + step),
              h(ProbPanel, { step: cur, trueLabel: tr.label, tt: probTip })),
            h("div", { className: "note" },
              "The final answer is removed from every partial input: during genuine runtime the agent has not yet answered, so a partial trace must not contain one.")))));
  }

  /* ── panel 3: results ──────────────────────────────────────────────────── */
  function ConfusionMatrix() {
    var tt = useTooltip();
    var off = D.offline;
    var order = off.confusionOrder, M = off.confusion;
    var max = 0;
    M.forEach(function (row) { row.forEach(function (v) { if (v > max) max = v; }); });
    var N = 7;

    // Snap the count to a step of the sequential ramp. Each step ships with a
    // paired ink token, so light and dark can re-step independently without the
    // label colour having to be recomputed here.
    function cell(v) {
      if (v === 0) return { bg: "var(--surface-2)", fg: "var(--text-3)" };
      var i = Math.min(N, Math.max(1, Math.round((v / max) * (N - 1)) + 1));
      return { bg: cvar("--seq-" + i), fg: cvar("--ink-" + i) };
    }

    return h("div", null,
      tt.node,
      h("div", { style: { overflowX: "auto" } },
        h("table", { style: { borderCollapse: "separate", borderSpacing: "2px",
                              fontVariantNumeric: "tabular-nums" } },
          h("thead", null,
            h("tr", null,
              h("th", { style: { width: "150px" } }),
              order.map(function (c) {
                return h("th", { key: c, style: {
                    fontSize: "11px", fontWeight: 600, color: "var(--text-3)",
                    padding: "0 0 6px", textAlign: "center", minWidth: "76px" } },
                  labelOf(c));
              }))),
          h("tbody", null,
            order.map(function (rc, ri) {
              return h("tr", { key: rc },
                h("td", { style: { fontSize: "12.5px", color: "var(--text-2)",
                                   paddingRight: "10px", whiteSpace: "nowrap" } },
                  h(Swatch, { color: colorOf(rc) }), labelOf(rc)),
                order.map(function (cc, ci) {
                  var v = M[ri][ci], st = cell(v);
                  var rowTotal = M[ri].reduce(function (a, b) { return a + b; }, 0);
                  return h("td", {
                    key: cc,
                    onMouseMove: function (e) {
                      tt.show(e, h("span", null,
                        h("strong", null, v),
                        ri === ci ? " correctly classified " : " misclassified ",
                        h("strong", null, labelOf(rc)),
                        ri === ci ? null : h("span", null, " as ", h("strong", null, labelOf(cc))),
                        h("span", { style: { color: "var(--text-3)" } },
                          "  ·  " + (rowTotal ? Math.round((v / rowTotal) * 100) : 0) + "% of true " + labelOf(rc))));
                    },
                    onMouseLeave: tt.hide,
                    style: { background: st.bg, color: st.fg, textAlign: "center",
                             padding: "13px 0", borderRadius: "4px", cursor: "default",
                             fontWeight: ri === ci ? 650 : 400, fontSize: "13px" }
                  }, v);
                }));
            })))),
      h("div", { className: "note" },
        "Rows are true labels, columns predictions. Eleven of the sixteen errors sit on the hallucination–success boundary; unsafe execution shows a clean column."));
  }

  function Results() {
    var off = D.offline, bl = D.baselines, rt = D.runtime;
    var best = {};
    Object.keys(bl.perClass).forEach(function (cls) {
      best[cls] = Math.max.apply(null, bl.perClass[cls]);
    });

    return h("div", null,
      h("section", { className: "block" },
        h("h2", null, "Per-class performance"),
        h("p", { className: "lede" },
          "DeBERTa-v3 on the held-out test set. Support is shown because the smaller classes carry proportionally less evidence."),
        h("div", { className: "grid g2" },
          h("div", { className: "card" },
            h("table", { className: "data" },
              h("thead", null, h("tr", null,
                h("th", null, "Class"), h("th", null, "Precision"),
                h("th", null, "Recall"), h("th", null, "F1"), h("th", null, "n"))),
              h("tbody", null,
                off.perClass.map(function (r) {
                  return h("tr", { key: r.name },
                    h("td", null, h(Swatch, { color: colorOf(r.name) }), labelOf(r.name)),
                    h("td", { className: "mono" }, f3(r.precision)),
                    h("td", { className: "mono" }, f3(r.recall)),
                    h("td", { className: "mono best" }, f3(r.f1)),
                    h("td", { className: "mono" }, r.support));
                }),
                h("tr", { className: "total" },
                  h("td", null, "Macro average"),
                  h("td", { className: "mono" }, "0.818"),
                  h("td", { className: "mono" }, "0.824"),
                  h("td", { className: "mono" }, off.macroF1.toFixed(3)),
                  h("td", { className: "mono" }, "87"))))),
          h("div", { className: "card" },
            h(ConfusionMatrix, null)))),

      h("section", { className: "block" },
        h("h2", null, "Model comparison"),
        h("p", { className: "lede" },
          "All five models trained and evaluated on the identical frozen splits. The advantage is not uniform: RoBERTa wins on loop, and the entire macro-F1 gap between the two transformers comes from unsafe execution."),
        h("div", { className: "card" },
          h("div", { style: { overflowX: "auto" } },
            h("table", { className: "data" },
              h("thead", null, h("tr", null,
                h("th", null, "Class"),
                bl.models.map(function (m) { return h("th", { key: m.short }, m.short); }))),
              h("tbody", null,
                ORDER.map(function (cls) {
                  var vals = bl.perClass[cls];
                  return h("tr", { key: cls },
                    h("td", null, h(Swatch, { color: colorOf(cls) }), labelOf(cls)),
                    vals.map(function (v, i) {
                      return h("td", { key: i,
                        className: "mono" + (v === best[cls] && v > 0 ? " best" : "") },
                        v.toFixed(3));
                    }));
                }),
                h("tr", { className: "total" },
                  h("td", null, "Macro F1"),
                  bl.models.map(function (m) {
                    return h("td", { key: m.short, className: "mono" }, m.macroF1.toFixed(3));
                  }))))),
          h("div", { className: "note" },
            "Bold marks the best model for each class. Single-seed runs at seed 42; a multi-seed mean would be the rigorous alternative."))),

      h("section", { className: "block" },
        h("h2", null, "Runtime detection"),
        h("p", { className: "lede" },
          "Each of the 61 failure traces in the test set was replayed one step at a time with the final answer withheld. Detection rate is the runtime measure: the failure was named before the agent finished."),
        h("div", { className: "grid g2" },
          h("div", { className: "card" },
            h("table", { className: "data" },
              h("thead", null, h("tr", null,
                h("th", null, "Failure type"), h("th", null, "Detected"),
                h("th", null, "Before final"), h("th", null, "Avg point"))),
              h("tbody", null,
                rt.byType.map(function (r) {
                  return h("tr", { key: r.name },
                    h("td", null, h(Swatch, { color: colorOf(r.name) }), labelOf(r.name)),
                    h("td", { className: "mono" }, r.detectedAny + " / " + r.n),
                    h("td", { className: "mono best" },
                      r.detectedBefore + " / " + r.n + "  (" + pct(r.rate) + ")"),
                    h("td", { className: "mono" },
                      r.avgPoint == null ? "—" : pct(r.avgPoint)));
                })))),
          h("div", { className: "card" },
            h("div", { style: { fontSize: "12px", textTransform: "uppercase",
                                letterSpacing: ".06em", color: "var(--text-3)",
                                fontWeight: 600, marginBottom: "14px" } },
              "Detected before the final step"),
            h(BarChart, {
              max: 1, labelWidth: 118,
              format: function (v) { return pct(v); },
              rows: rt.byType.map(function (r) {
                return { name: labelOf(r.name), value: r.rate, color: colorOf(r.name) };
              })
            }),
            h("div", { className: "note" },
              "Unsafe execution is caught fastest — a single visible action event is enough. Loop is caught reliably but late, because repetition is not visible until it has occurred. Hallucination is the limiting case: the ungrounded claim frequently exists only in the final answer.")))));
  }

  /* ── shell ─────────────────────────────────────────────────────────────── */
  var TABS = [
    { id: "overview", label: "Overview",     C: Overview },
    { id: "replay",   label: "Trace replay", C: Replay },
    { id: "results",  label: "Results",      C: Results }
  ];

  function App() {
    var [tab, setTab] = useState("overview");
    var [theme, setTheme] = useState(function () {
      try { return localStorage.getItem("afd-theme") || "auto"; } catch (e) { return "auto"; }
    });

    useEffect(function () {
      if (theme === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", theme);
      try { localStorage.setItem("afd-theme", theme); } catch (e) {}
    }, [theme]);

    var Cur = (TABS.filter(function (t) { return t.id === tab; })[0] || TABS[0]).C;

    return h(React.Fragment, null,
      h("header", { className: "top" },
        h("div", { className: "top-inner" },
          h("div", { className: "brand" },
            h("h1", null, "Runtime Detection of Failure Modes in LLM Agents"),
            h("span", { className: "sub" },
              D.meta.model + " · " + D.meta.agent),
            h("span", { className: "spacer" }),
            h("button", {
              className: "themebtn",
              onClick: function () {
                setTheme(theme === "auto" ? "light" : theme === "light" ? "dark" : "auto");
              }
            }, "Theme: " + theme)),
          h("nav", { className: "tabs" },
            TABS.map(function (t) {
              return h("button", {
                key: t.id, "aria-selected": t.id === tab,
                onClick: function () { setTab(t.id); }
              }, t.label);
            })))),
      h("main", { className: "shell" },
        h(Cur, null),
        h("footer", { className: "foot" },
          h("span", null, "Static viewer — no inference is run in the browser."),
          h("span", null, "Data: dashboard_data.js, generated from the frozen evaluation artefacts."),
          h("span", null, "Shaheer Aslam · MSc Artificial Intelligence · London South Bank University"))));
  }

  if (!D) {
    document.getElementById("root").innerHTML =
      '<div style="padding:40px;font-family:sans-serif">' +
      '<h2>dashboard_data.js not found</h2>' +
      '<p>Run <code>python demo/export_dashboard_data.py</code> from the project root first.</p></div>';
  } else {
    ReactDOM.createRoot(document.getElementById("root")).render(h(App, null));
  }
})();
