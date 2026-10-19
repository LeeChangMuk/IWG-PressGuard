"""Reproduce PressGuard Light v1.2 from the two KAMP CSV files.

Usage: python train.py --data-dir data --output-dir ../evidence [--model-path pumpguard/model.json]

Order of operations (fixed rules live in protocol.py):
 1. load, de-duplicate, cut segments at gaps > 0.15 s, assign the fixed test split (last 20 %)
 2. blocked & purged 5-fold CV inside development -> out-of-fold (OOF) scores for 8 candidates
 3. per candidate: Platt calibration on OOF, standard threshold on OOF, OOF metrics
 4. selection by the fixed rule -> selection_protocol.json is written BEFORE any test metric
 5. final model = selected candidate refit on all judged development rows, exported to JSON
 6. single test evaluation: rows, 2-of-3 persistent alarm (deployed engine), segments, bootstrap
 7. report-only: every candidate on test, v1.1 reference model on test
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
from protocol import (SEED, CANDIDATES, CANDIDATE_BY_NAME, CENTERED, MIN_HISTORY, FPR_CAP, DEV_FOLDS,
                      PURGE_SEGMENTS, TEST_SHARE, MODE_RULES, SELECTION_RULE, load_dataset, make_features,
                      blocked_folds, oof_scores, fit_candidate, fit_platt, apply_platt, threshold_fpr_cap, mode_thresholds,
                      row_metrics, persistent_alarm, alarm_metrics, segment_bootstrap, select_candidate, export_predictor,
                      write_json, korean_font)
from pumpguard.features import CHANNELS, FEATURE_NAMES, FEATURE_VERSION
from pumpguard.inference import predict_features, StreamAnalyzer, validate_model

MODES = ("unmanned", "standard", "screening")
MODE_KO = {"unmanned": "무인 감시", "standard": "표준 운전", "screening": "점검 선별"}


def evaluate_oof(name, raw, y, dev, judged, folds, segments):
    fit_rows = dev & judged
    cal = fit_platt(raw[fit_rows], y[fit_rows])
    p = np.full(len(y), np.nan)
    p[dev] = apply_platt(raw[dev], cal)
    thr = threshold_fpr_cap(y[fit_rows], p[fit_rows], FPR_CAP["standard"])
    overall = row_metrics(y[dev], p[dev], judged[dev], thr)
    fold_f1, fold_ap = [], []
    for _, te in folds:
        m = row_metrics(y[te], p[te], judged[te], thr)
        fold_f1.append(m["f1"]); fold_ap.append(m.get("ap_judged", np.nan))
    pred = judged[dev] & (p[dev] >= thr)
    alarm, event = persistent_alarm(pred, segments[dev], judged[dev])
    persistent = alarm_metrics(y[dev], alarm, event, segments[dev])
    return {"name": name, "calibration": cal, "threshold_standard": thr, "row": overall,
            "f1": overall["f1"], "ap": overall["ap_judged"], "fp": overall["fp"],
            "fold_f1": fold_f1, "fold_ap": fold_ap, "persistent": persistent}, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="../evidence")
    parser.add_argument("--model-path", help="Optional deployment-model destination; omitted leaves the bundled model untouched")
    parser.add_argument("--reference-model", help="Optional v1.1 model.json, evaluated on test for continuity only")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    korean_font()

    d, sources = load_dataset(args.data_dir)
    X, history = make_features(d)
    d["n_history"] = history
    y = d.label.to_numpy()
    judged = history >= MIN_HISTORY
    dev = d.split.eq("development").to_numpy(); test = ~dev
    segments = d.segment_id.to_numpy()
    Xc = X[:, CENTERED]
    assert np.isfinite(X).all()
    folds = blocked_folds(d, dev)
    fold_of = np.full(len(d), -1)
    purged_in = {}
    for k, (tr, te) in enumerate(folds):
        fold_of[te] = k
        purged_in[k] = sorted(set(segments[dev & ~tr & ~te]))
    assert (fold_of[dev] >= 0).all() and (fold_of[test] == -1).all()

    manifest = d.groupby("segment_id", sort=False).agg(source_file=("source_file", "first"), label=("label", "first"),
                                                       split=("split", "first"), dev_block=("dev_block", "first"),
                                                       start=("TimeStamp", "first"), end=("TimeStamp", "last"),
                                                       rows=("TimeStamp", "size")).reset_index()
    manifest.to_csv(out / "split_manifest.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- development OOF for every candidate
    print("Development blocked CV (test unused) ...", flush=True)
    results, oof_p = {}, {}
    for c in CANDIDATES:
        t0 = time.perf_counter()
        raw = oof_scores(c["name"], Xc, y, folds, judged)
        res, p = evaluate_oof(c["name"], raw, y, dev, judged, folds, segments)
        res["seconds"] = time.perf_counter() - t0
        results[c["name"]], oof_p[c["name"]] = res, p
        print(f"  {c['name']:17s} OOF F1 {res['f1']:.4f} AP {res['ap']:.4f} FP {res['fp']}", flush=True)
    selected = select_candidate(results)
    sel = results[selected]
    protocol = {"version": "1.2.0", "seed": SEED,
                "split": f"Within each file, time-ordered segments; last {TEST_SHARE:.0%} = test (same boundary as v1.0); rest = development",
                "development_cv": f"{DEV_FOLDS} contiguous segment blocks per file, purge {PURGE_SEGMENTS} neighbouring segment(s) on each side",
                "features": "11 offset-invariant causal features (centered); level features excluded by design (label = file = date)",
                "judged_rows": f"n_history >= {MIN_HISTORY}; warm-up rows never predict, vote or alarm and count as no-alarm",
                "training_rows": "judged rows only; class_weight=balanced for supervised candidates",
                "calibration": "Platt scaling fitted on development OOF raw scores of judged rows",
                "thresholds": MODE_RULES, "selection_rule": SELECTION_RULE,
                "alarm_policy": {"window": 3, "votes_required": 2, "minimum_history": MIN_HISTORY, "votes_from": "judged rows only"},
                "candidates": CANDIDATES, "selected": selected, "test_used_for_selection": False,
                "development_oof": {n: {"f1": r["f1"], "ap": r["ap"], "fp": r["fp"], "threshold_standard": r["threshold_standard"]}
                                    for n, r in results.items()},
                "sources": sources}
    write_json(out / "selection_protocol.json", protocol)
    print("Selected:", selected, flush=True)
    # ----- selection frozen. Everything below fits the frozen choice or reports.

    fit_rows = dev & judged
    final = fit_candidate(selected, Xc[fit_rows], y[fit_rows])
    predictor, indices = export_predictor(final)
    thresholds = mode_thresholds(y[fit_rows], oof_p[selected][fit_rows])
    assert abs(thresholds["standard"] - sel["threshold_standard"]) < 1e-15
    dev_values = d.loc[dev, list(CHANNELS)].to_numpy()
    normal_dev = dev & (y == 0)
    limits = np.maximum(np.max(np.abs(dev_values), axis=0) * 10, np.std(dev_values, axis=0) * 50)
    model = {"schema_version": 2, "model_version": "1.2.0", "model_id": "IWG-PressGuard-Light-" + selected,
             "selected_candidate": selected, "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
             "channels": list(CHANNELS), "feature_indices": indices, "predictor": predictor, "calibration": sel["calibration"],
             "threshold": thresholds["standard"],
             "operating_modes": {k: {"threshold": thresholds[k], "rule": MODE_RULES[k]} for k in MODES},
             "alarm_policy": {"window": 3, "votes_required": 2, "minimum_history": MIN_HISTORY},
             "max_window": 10, "max_gap_seconds": .15, "expected_sampling_hz": 10., "expected_sample_interval_seconds": .1,
             "review_absolute_limits": limits.tolist(), "training_seed": SEED, "field_validation": "NOT_PERFORMED",
             "scope": "Dataset state classification with calibrated in-dataset probability; not a field failure probability or lead time",
             "input_units": "Source units unknown; site mapping and baseline confirmation required",
             "normal_reference": {"feature_p99": {n: float(v) for n, v in zip(FEATURE_NAMES, np.quantile(X[normal_dev & judged], .99, axis=0))},
                                  "raw_median": np.median(d.loc[normal_dev, list(CHANNELS)].to_numpy(), axis=0).tolist()}}
    validate_model(model)
    write_json(out / "model.json", model)
    if args.model_path:
        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.model_path, model)
    model_bytes = (out / "model.json").stat().st_size

    # ---------------------------------------------------------------- single test evaluation
    td = d.loc[test].copy().reset_index(drop=True)
    ty, tj, tseg = y[test], judged[test], segments[test]
    p_test = apply_platt(final.raw(Xc[test]), sel["calibration"])
    exported = np.array([predict_features(row.tolist(), model) for row in X[test]])
    export_error = float(np.max(np.abs(exported - p_test)))
    if export_error > 1e-9:
        raise AssertionError(f"JSON export mismatch {export_error}")
    test_rows = {k: row_metrics(ty, p_test, tj, thresholds[k]) for k in MODES}

    engine, records = {}, None
    for mode in MODES:
        recs = []
        for _, g in td.groupby("source_file", sort=False):
            analyzer = StreamAnalyzer(model, mode)
            for row in g.itertuples():
                recs.append(analyzer.update([getattr(row, c) for c in CHANNELS], row.TimeStamp))
        rec = pd.DataFrame(recs)
        if not np.allclose(rec.score.to_numpy(dtype=float), p_test, atol=1e-9):
            raise AssertionError("Deployed engine score differs")
        sim_pred = tj & (p_test >= thresholds[mode])
        if not np.array_equal(rec.predicted.to_numpy(bool), sim_pred):
            raise AssertionError("Deployed engine decision differs")
        sim_alarm, sim_event = persistent_alarm(sim_pred, tseg, tj)
        if not np.array_equal(rec.alarm.to_numpy(bool), sim_alarm):
            raise AssertionError("Deployed engine persistence differs")
        engine[mode] = alarm_metrics(ty, sim_alarm, sim_event, tseg)
        if mode == "standard":
            records = rec
    segs = []
    for sid, g in td.assign(alarm=records.alarm.to_numpy(bool), score=p_test, judged=tj).groupby("segment_id", sort=False):
        first = g[g.alarm]
        segs.append({"segment_id": sid, "label": int(g.label.iloc[0]), "rows": len(g), "judged_rows": int(g.judged.sum()),
                     "mean_score": float(g.score[g.judged].mean()) if g.judged.any() else None,
                     "alarm_rows": int(g.alarm.sum()), "detected": bool(len(first)),
                     "observed_start_to_first_alarm_s": float((pd.to_datetime(first.TimeStamp.iloc[0]) - pd.to_datetime(g.TimeStamp.iloc[0])).total_seconds()) if len(first) else None})
    pd.DataFrame(segs).to_csv(out / "test_segment_alarms.csv", index=False, encoding="utf-8-sig")
    boot = segment_bootstrap(ty, p_test, tj, tseg, thresholds["standard"])

    pred_out = pd.DataFrame({"source_file": td.source_file, "original_csv_line": td.source_row + 2, "segment_id": td.segment_id,
                             "TimeStamp": td.TimeStamp, **{c: td[c] for c in CHANNELS}, "label": ty, "n_history": td.n_history,
                             "judged": tj.astype(int), "score": p_test,
                             **{f"decision_{k}": (tj & (p_test >= thresholds[k])).astype(int) for k in MODES},
                             "alarm_standard": records.alarm.astype(int), "status_standard": records.status, "reason_standard": records.reason})
    pred_out.to_csv(out / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- report-only comparisons on test
    candidate_test = {}
    for c in CANDIDATES:
        name = c["name"]
        fitted = final if name == selected else fit_candidate(name, Xc[fit_rows], y[fit_rows])
        pt = apply_platt(fitted.raw(Xc[test]), results[name]["calibration"])
        m = row_metrics(ty, pt, tj, results[name]["threshold_standard"])
        a, e = persistent_alarm(tj & (pt >= results[name]["threshold_standard"]), tseg, tj)
        candidate_test[name] = {"row": m, "persistent": alarm_metrics(ty, a, e, tseg)}
    reference = None
    if args.reference_model:
        ref = json.loads(Path(args.reference_model).read_text(encoding="utf-8"))
        pr = np.array([predict_features(row.tolist(), ref) for row in X[test]])
        rm = row_metrics(ty, pr, tj, ref["threshold"])
        a, e = persistent_alarm(tj & (pr >= ref["threshold"]), tseg, tj)
        legacy_tp = int(np.sum((pr >= ref["threshold"]) & (ty == 1))); legacy_fp = int(np.sum((pr >= ref["threshold"]) & (ty == 0)))
        legacy_fn = int(np.sum(ty == 1)) - legacy_tp
        reference = {"model_id": ref.get("model_id"), "model_version": ref.get("model_version"), "threshold": ref["threshold"],
                     "row_judged_rule": rm, "persistent": alarm_metrics(ty, a, e, tseg),
                     "row_v1_rule_all_rows": {"tp": legacy_tp, "fp": legacy_fp, "fn": legacy_fn,
                                              "f1": 2 * legacy_tp / (2 * legacy_tp + legacy_fp + legacy_fn)},
                     "note": "Frozen v1.1 predictor (ExtraTrees, trained on 50% only). Re-scored with the v1.2 judged-row rule; report only."}

    # ---------------------------------------------------------------- outputs
    oof_table = d.loc[dev, ["segment_id", "source_file", "source_row", "TimeStamp", "label", "n_history"]].copy()
    oof_table["judged"] = judged[dev].astype(int); oof_table["fold"] = fold_of[dev]
    for c in CANDIDATES:
        oof_table["oof_" + c["name"]] = oof_p[c["name"]][dev]
    oof_table.to_csv(out / "oof_development.csv", index=False, encoding="utf-8-sig")

    comparison = []
    for c in CANDIDATES:
        r, t = results[c["name"]], candidate_test[c["name"]]
        comparison.append({"name": c["name"], "label": c["label"], "family": c["family"], "eligible": c["eligible"],
                           "oof_f1": r["f1"], "oof_precision": r["row"]["precision"], "oof_recall": r["row"]["recall"],
                           "oof_ap": r["ap"], "oof_fp": r["fp"], "oof_fold_f1_mean": float(np.mean(r["fold_f1"])),
                           "oof_fold_f1_min": float(np.min(r["fold_f1"])), "oof_fold_f1_max": float(np.max(r["fold_f1"])),
                           "oof_fold_f1_sd": float(np.std(r["fold_f1"], ddof=1)), "oof_persistent_f1": r["persistent"]["f1"],
                           "test_f1": t["row"]["f1"], "test_precision": t["row"]["precision"], "test_recall": t["row"]["recall"],
                           "test_ap": t["row"].get("ap_judged"), "test_fp": t["row"]["fp"], "test_fn": t["row"]["fn"],
                           "test_persistent_f1": t["persistent"]["f1"], "selected": c["name"] == selected})
    pd.DataFrame(comparison).to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")

    report = {"project": "IWG 프레스가드 라이트", "model_version": "1.2.0", "selected_model": selected,
              "selection_protocol": protocol,
              "data": {"sources": sources, "rows": int(len(d)), "judged_rows": int(judged.sum()),
                       "split_rows": d.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
                       "split_judged_rows": d[judged].groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
                       "split_segments": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
                       "dev_fold_segments": {int(k): {"anomalous": int(len(set(segments[te & (y == 1)]))), "normal": int(len(set(segments[te & (y == 0)]))),
                                                      "purged": purged_in[k]} for k, (tr, te) in enumerate(folds)},
                       "selected_feature_names": [FEATURE_NAMES[i] for i in indices]},
              "development_oof": results, "thresholds": thresholds,
              "development_oof_selected_modes": {k: row_metrics(y[dev], oof_p[selected][dev], judged[dev], thresholds[k]) for k in MODES},
              "test": {"rows": test_rows, "persistent": engine, "segment_bootstrap_standard": boot,
                       "anomalous_segment_detail": [s for s in segs if s["label"] == 1]},
              "candidates_test_report_only": candidate_test, "reference_v11_test_report_only": reference,
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "sklearn": sklearn.__version__,
                          "model_json_bytes": model_bytes, "json_export_max_abs_error": export_error,
                          "training_and_evaluation_seconds": time.perf_counter() - started},
              "limitations": ["Only two source recordings: label = file = date; cross-equipment / cross-date validation impossible.",
                              "No fault onset or maintenance history: early-warning lead time cannot be measured.",
                              "Five anomalous test segments from one 2.8-minute recording; intervals are conditional on these files.",
                              "Units, product mix, speed/load and maintenance records absent.",
                              "Calibrated probabilities reflect the in-dataset anomaly share (2.9%); re-calibrate for field prevalence."]}
    write_json(out / "metrics.json", report)
    make_plots(out, d, y, judged, dev, test, td, p_test, thresholds, results, candidate_test, selected, folds)
    write_summary(out, report, comparison, selected, thresholds, test_rows, engine, boot, reference)
    print(json.dumps({"selected": selected, "test_standard": test_rows["standard"], "persistent_standard": engine["standard"]},
                     ensure_ascii=False), flush=True)


def make_plots(out, d, y, judged, dev, test, td, p_test, thresholds, results, candidate_test, selected, folds):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    # 1) validation design: segment timeline per file
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.8))
    blues = ["#3f6f96", "#5b8db8", "#7aa7cc", "#97bfdd", "#b5d3ea"]
    seg_all = d.segment_id.to_numpy()
    for ax, (f, title) in zip(axes, [("press_data_normal.csv", "정상 파일(7/12): 연속구간 599개"),
                                      ("outlier_data.csv", "이상 파일(7/17): 연속구간 21개")]):
        g = d[d.source_file.eq(f)].drop_duplicates("segment_id")
        cols = ["#c0392b" if s == "test" else blues[b % 5] for s, b in zip(g.split, g.dev_block)]
        ax.bar(g.segment_order, 1, width=1.0, color=cols, edgecolor="white" if f.startswith("outlier") else None, linewidth=.8)
        filemask = d.source_file.eq(f).to_numpy()
        for tr, te in folds:
            for sid in set(seg_all[dev & ~tr & ~te & filemask]):
                o = int(g.loc[g.segment_id.eq(sid), "segment_order"].iloc[0])
                ax.plot([o], [1.13], marker="v", color="#333", ms=3 if f.startswith("press") else 6)
        ax.set_yticks([]); ax.set_ylim(0, 1.32); ax.set_xlim(-1, g.segment_order.max() + 1)
        if f.startswith("outlier"):
            ax.set_xticks(range(0, 21, 2))
        ax.set_title(title, fontsize=10, loc="left")
        for k in range(5):
            sub = g[g.dev_block.eq(k) & g.split.eq("development")]
            if len(sub):
                ax.text(sub.segment_order.mean(), .5, f"개발{k + 1}", ha="center", va="center", fontsize=8, color="white")
        tsub = g[g.split.eq("test")]
        ax.text(tsub.segment_order.mean(), .5, "시험", ha="center", va="center", fontsize=9, color="white")
    axes[1].set_xlabel("시간 순서(연속구간 번호)   ▼ = 해당 블록 평가 시 학습에서 뺀 인접 구간(퍼지)")
    fig.suptitle("검증 설계: 개발 80%는 시간 블록 5-겹 교차검증(퍼지 1구간), 마지막 20%는 최종 1회 시험", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "validation_design.png"); plt.close(fig)

    # 2) model comparison on development OOF (+ test report-only markers)
    names = [c["name"] for c in CANDIDATES]
    labels = [c["label"].split("(")[0] for c in CANDIDATES]
    order = np.argsort([results[n]["f1"] for n in names])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for pos, i in enumerate(order):
        n = names[i]; r = results[n]
        color = "#087f8c" if n == selected else ("#c4ccd3" if not CANDIDATES[i]["eligible"] else "#7d93a6")
        ax.barh(pos, r["f1"], color=color, height=.62)
        ax.plot([min(r["fold_f1"]), max(r["fold_f1"])], [pos, pos], color="#222", lw=1.4)
        ax.plot(candidate_test[n]["row"]["f1"], pos, marker="D", color="#e76f51", ms=6, mfc="white", mew=1.6)
        ax.text(r["f1"] + .012, pos + .27, f"{r['f1']:.3f}", fontsize=8, va="center")
    ax.set_yticks(range(len(names)), [labels[i] + (" ★" if names[i] == selected else "") for i in order])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 — 막대: 개발 OOF 전체, 검은 선: 블록별 최소~최대, ◇: 시험(보고 전용, 선정에 미사용)")
    ax.set_title("후보 8개 비교 (선정은 개발 구간 블록 교차검증 OOF F1로만 수행)")
    fig.tight_layout(); fig.savefig(out / "model_comparison.png"); plt.close(fig)

    # 3) test performance: confusion matrix (standard rows) + PR curve with mode points
    ty, tj = y[test], judged[test]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    pred = tj & (p_test >= thresholds["standard"])
    cm = confusion_matrix(ty, pred, labels=[0, 1])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=17, color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xticks([0, 1], ["정상 판정", "이상 판정"]); axes[0].set_yticks([0, 1], ["실제 정상", "실제 이상"])
    axes[0].set_title("시험 혼동행렬 (표준 모드, 표본 판정)")
    # warm-up rows can never be flagged: give them the lowest score so recall matches the reported definition
    curve_score = np.where(tj, p_test, -1.0)
    pr, rc, th = precision_recall_curve(ty, curve_score)
    keep = np.r_[th > -1.0, True]
    axes[1].plot(rc[keep], pr[keep], color="#087f8c", label="선정 모델(준비 행은 미탐으로 계산)")
    axes[1].axhline(ty.mean(), ls="--", color="grey", label=f"무작위 기준 {ty.mean():.3f}")
    for k, mk in [("unmanned", "s"), ("standard", "o"), ("screening", "^")]:
        m = row_metrics(ty, p_test, tj, thresholds[k])
        axes[1].plot(m["recall"], m["precision"], marker=mk, ms=8, color="#e76f51", ls="none", label=f"{MODE_KO[k]} (기준 {thresholds[k]:.3f})")
    axes[1].set(xlabel="재현율", ylabel="정밀도", title="시험 정밀도-재현율 곡선과 운영 모드", xlim=(0, 1.01), ylim=(0, 1.03))
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "test_performance.png"); plt.close(fig)

    # 4) anomalous test segments timeline
    a = td.label.eq(1).to_numpy(); da = td[a].reset_index(drop=True); pa = p_test[a]; xx = np.arange(len(da))
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True)
    for c, col in zip(CHANNELS[:2], ["#1f77b4", "#ff7f0e"]):
        axs[0].plot(xx, da[c], lw=.8, label=c, color=col)
    axs[0].legend(loc="upper right", fontsize=8); axs[0].set_ylabel("진동(원 단위)")
    axs[1].plot(xx, da.AI2_Current, color="#62758a", lw=.8); axs[1].set_ylabel("전류(원 단위)")
    jj = da.n_history.to_numpy() >= MIN_HISTORY
    axs[2].plot(xx[jj], pa[jj], ".", ms=4, color="#087f8c", label="보정 이상확률(판정 행)")
    for k, ls in [("unmanned", "-."), ("standard", "--"), ("screening", ":")]:
        axs[2].axhline(thresholds[k], ls=ls, color="#e76f51", lw=1, label=f"{MODE_KO[k]} 기준")
    axs[2].set(ylim=(-.02, 1.03), ylabel="이상확률", xlabel="시험 이상 표본 순서 (점선 = 연속구간 경계)")
    axs[2].legend(fontsize=7, loc="center right")
    bounds = np.flatnonzero(da.segment_id.to_numpy()[1:] != da.segment_id.to_numpy()[:-1]) + 1
    for ax in axs:
        for b in bounds:
            ax.axvline(b, color="grey", alpha=.5, ls=":")
    top = axs[0].get_ylim()[1]
    for sid, g in da.groupby("segment_id", sort=False):
        axs[0].text(g.index.to_numpy().mean(), top * .8, sid, ha="center", fontsize=8)
    fig.suptitle("시험 이상 연속구간 5개: 신호와 보정 이상확률", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "test_anomaly_timeline.png"); plt.close(fig)


def write_summary(out, report, comparison, selected, thresholds, test_rows, engine, boot, reference):
    m = test_rows["standard"]; e = engine["standard"]; b = boot["bounds95"]
    lines = ["# 프레스가드 라이트 v1.2 결과 요약", "",
             f"선정 모델: **{selected}** ({CANDIDATE_BY_NAME[selected]['label']}). 선정·보정·임계값은 개발 구간 블록 교차검증 OOF만 사용했다.", "",
             "## 개발 OOF 후보 비교 (선정 근거)", "|후보|구분|OOF F1|OOF 정밀도|OOF 재현율|OOF AP|블록 F1 최소~최대|시험 F1(보고 전용)|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(comparison, key=lambda r: -r["oof_f1"]):
        lines.append(f"|{c['name']}{' ★' if c['selected'] else ''}|{c['family']}|{c['oof_f1']:.3f}|{c['oof_precision']:.3f}|{c['oof_recall']:.3f}|"
                     f"{c['oof_ap']:.3f}|{c['oof_fold_f1_min']:.3f}~{c['oof_fold_f1_max']:.3f}|{c['test_f1']:.3f}|")
    lines += ["", "## 시험 1회 평가 (표준 모드)",
              f"- 표본 판정: F1 {m['f1']:.4f}, 정밀도 {m['precision']:.4f}, 재현율 {m['recall']:.4f}, AP {m['ap_judged']:.4f}, ROC-AUC {m['roc_auc_judged']:.4f}, Brier {m['brier_judged']:.5f}",
              f"- TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']} (판정 대상 {m['judged_rows']:,}행 / 전체 {m['rows']:,}행; 준비 행은 무경보로 계산)",
              f"- 지속성 경보(3표본 중 2): F1 {e['f1']:.4f}, 정밀도 {e['precision']:.4f}, 재현율 {e['recall']:.4f}, FP {e['fp']}, 이상 구간 {e['anomalous_segments_detected']}/{e['anomalous_segments_total']} 포착, 정상 {e['normal_observed_seconds']:.0f}초 중 경보 진입 {e['normal_alarm_events']}건",
              f"- 구간 부트스트랩 95%: F1 {b['f1'][0]:.3f}~{b['f1'][1]:.3f}, AP {b['ap_judged'][0]:.3f}~{b['ap_judged'][1]:.3f} (이상 구간 5개 조건부)", "",
              "## 운영 모드 (개발 OOF 규칙)", "|모드|임계값|시험 정밀도|시험 재현율|시험 F1|지속경보 FP|이상 구간 포착|", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in MODES:
        r, en = test_rows[k], engine[k]
        lines.append(f"|{MODE_KO[k]}|{thresholds[k]:.4f}|{r['precision']:.3f}|{r['recall']:.3f}|{r['f1']:.3f}|{en['fp']}|{en['anomalous_segments_detected']}/{en['anomalous_segments_total']}|")
    if reference:
        rr = reference["row_judged_rule"]
        lines += ["", "## v1.1 모델(ExtraTrees) 참고 비교 (보고 전용)",
                  f"- 같은 판정 규칙: F1 {rr['f1']:.4f} (정밀도 {rr['precision']:.3f}, 재현율 {rr['recall']:.3f}); v1.1 원래 규칙(준비 행 포함) F1 {reference['row_v1_rule_all_rows']['f1']:.4f}"]
    lines += ["", "## 한계", *[f"- {t}" for t in report["limitations"]], "",
              "## 재현", "`python reproduce.py` (원본 해시 확인 → 학습·보정·선정·시험·분석·그림 일괄 실행)"]
    (out / "MODEL_RESULTS.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
