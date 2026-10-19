"""Nested, blocked cross-validation of the WHOLE v1.2 procedure over all 21 anomalous segments.

Usage: python nested_cv.py --data-dir data --output-dir ../evidence

Outer: 7 contiguous segment blocks per file over the full recording (purge 1 neighbour).
Inner: the exact development protocol (5 blocked folds, purge 1) on the outer-training part:
       OOF scores for all 8 candidates -> Platt -> standard threshold -> selection rule.
The selected candidate is refit on the outer-training judged rows and scored once on the
outer block. Every candidate is also scored the same way (with its own inner calibration and
threshold) so the comparison itself carries an uncertainty estimate.
Purpose: estimate how much the reported performance and the model choice depend on which
anomalous segments happen to be held out. It is NOT used to choose anything.
Note: outer blocks train on data both before and after the held-out block.
"""
from pathlib import Path
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from protocol import (CANDIDATES, CENTERED, MIN_HISTORY, FPR_CAP, load_dataset, make_features, blocked_folds, oof_scores,
                      fit_candidate, fit_platt, apply_platt, threshold_fpr_cap, row_metrics, persistent_alarm, alarm_metrics,
                      select_candidate, write_json, korean_font)

warnings.filterwarnings("ignore")
OUTER_FOLDS = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="../evidence")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    korean_font()
    d, _ = load_dataset(args.data_dir)
    X, history = make_features(d)
    y = d.label.to_numpy(); judged = history >= MIN_HISTORY; seg = d.segment_id.to_numpy()
    Xc = X[:, CENTERED]
    everything = np.ones(len(d), bool)
    outer = blocked_folds(d, everything, k=OUTER_FOLDS, purge=1)
    rows, folds_out = [], []
    for k, (otr, ote) in enumerate(outer):
        t0 = time.perf_counter()
        inner = blocked_folds(d, otr, k=5, purge=1)
        inner_res = {}
        for c in CANDIDATES:
            raw = oof_scores(c["name"], Xc, y, inner, judged)
            fit = otr & judged
            cal = fit_platt(raw[fit], y[fit])
            p = apply_platt(raw[otr], cal)
            thr = threshold_fpr_cap(y[fit], apply_platt(raw[fit], cal), FPR_CAP["standard"])
            m = row_metrics(y[otr], p, judged[otr], thr)
            inner_res[c["name"]] = {"f1": m["f1"], "ap": m["ap_judged"], "fp": m["fp"], "cal": cal, "thr": thr}
        chosen = select_candidate(inner_res)
        for c in CANDIDATES:
            r = inner_res[c["name"]]
            model = fit_candidate(c["name"], Xc[otr & judged], y[otr & judged])
            p = apply_platt(model.raw(Xc[ote]), r["cal"])
            m = row_metrics(y[ote], p, judged[ote], r["thr"])
            a, e = persistent_alarm(judged[ote] & (p >= r["thr"]), seg[ote], judged[ote])
            am = alarm_metrics(y[ote], a, e, seg[ote])
            rows.append({"outer_fold": k + 1, "candidate": c["name"], "selected_by_inner_rule": c["name"] == chosen,
                         "inner_oof_f1": r["f1"], "outer_f1": m["f1"], "outer_precision": m["precision"], "outer_recall": m["recall"],
                         "outer_ap": m.get("ap_judged"), "outer_fp": m["fp"], "outer_fn": m["fn"], "outer_persistent_f1": am["f1"],
                         "outer_anomalous_segments": am["anomalous_segments_total"], "outer_segments_detected": am["anomalous_segments_detected"],
                         "outer_normal_alarm_events": am["normal_alarm_events"]})
        held = sorted(set(seg[ote & (y == 1)]))
        folds_out.append({"outer_fold": k + 1, "selected": chosen, "anomalous_segments_held_out": held,
                          "seconds": time.perf_counter() - t0})
        print(f"outer {k + 1}: selected {chosen}; held-out anomalous {held}; {time.perf_counter() - t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(out / "nested_cv.csv", index=False, encoding="utf-8-sig")
    procedure = r[r.selected_by_inner_rule]
    summary = {"outer_folds": OUTER_FOLDS, "folds": folds_out,
               "selection_counts": procedure.candidate.value_counts().to_dict(),
               "procedure": {k: {"mean": float(procedure[k].mean()), "min": float(procedure[k].min()), "max": float(procedure[k].max()),
                                 "sd": float(procedure[k].std(ddof=1))} for k in ("outer_f1", "outer_ap", "outer_persistent_f1", "outer_precision", "outer_recall")},
               "procedure_segments_detected": f"{int(procedure.outer_segments_detected.sum())}/{int(procedure.outer_anomalous_segments.sum())}",
               "per_candidate": {n: {"outer_f1_mean": float(g.outer_f1.mean()), "outer_f1_min": float(g.outer_f1.min()),
                                     "outer_f1_max": float(g.outer_f1.max()), "outer_ap_mean": float(g.outer_ap.mean()),
                                     "outer_persistent_f1_mean": float(g.outer_persistent_f1.mean()),
                                     "folds_ranked_first_by_outer_f1": int(sum(r[r.outer_fold == f].sort_values("outer_f1", ascending=False).candidate.iloc[0] == n
                                                                              for f in r.outer_fold.unique())),
                                     "segments_detected": f"{int(g.outer_segments_detected.sum())}/{int(g.outer_anomalous_segments.sum())}"}
                                 for n, g in r.groupby("candidate", sort=False)},
               "note": "Outer folds cover all 21 anomalous segments once. Estimates conditional on the two source recordings."}
    write_json(out / "nested_cv.json", summary)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    names = [c["name"] for c in CANDIDATES]; labels = [c["label"].split("(")[0] for c in CANDIDATES]
    means = [summary["per_candidate"][n]["outer_f1_mean"] for n in names]; order = np.argsort(means)
    for pos, i in enumerate(order):
        g = r[r.candidate == names[i]]
        ax.scatter(g.outer_f1, np.full(len(g), pos) + np.random.default_rng(i).uniform(-.12, .12, len(g)), s=18, color="#7d93a6", alpha=.8)
        ax.plot(means[i], pos, marker="|", ms=22, mew=3, color="#087f8c" if names[i] == "hist_gb" else "#333")
        sel = g[g.selected_by_inner_rule]
        ax.scatter(sel.outer_f1, np.full(len(sel), pos), s=70, facecolor="none", edgecolor="#e76f51", lw=1.6)
    ax.set_yticks(range(len(names)), [labels[i] for i in order])
    ax.set_xlim(0, 1); ax.set_xlabel("외부 폴드 F1 (점 = 7개 외부 폴드, 세로선 = 평균, 주황 원 = 내부 규칙이 그 폴드에서 선택한 모델)")
    ax.set_title(f"중첩 블록 교차검증: 선정 절차 전체를 7번 반복 (선택 횟수 {summary['selection_counts']})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "nested_cv.png", dpi=150); plt.close(fig)
    print(summary["selection_counts"], summary["procedure"]["outer_f1"], flush=True)


if __name__ == "__main__":
    main()
