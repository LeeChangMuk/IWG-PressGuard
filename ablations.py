"""Development-only design checks for the report (never uses the test split, never changes the model).

Usage: python ablations.py --data-dir data --output-dir ../evidence

1. Feature family: offset-invariant 11 vs all 27 (incl. level features) vs vibration window means only
   (a deliberate "shortcut detector": if file-level offsets carry the label, these alone will score well).
2. Imbalance handling: class_weight balanced vs none.
3. Training rows: judged rows only vs all rows (warm-up included).
4. Boosting hyper-parameter sensitivity, one factor at a time (reported, not tuned).
5. Calibration method, cross-fitted on OOF scores: none vs Platt vs isotonic.
All numbers: blocked 5-fold OOF on the development split, the same rules as train.py.
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from protocol import (CENTERED, LEVEL27, MIN_HISTORY, FPR_CAP, load_dataset, make_features, blocked_folds, oof_scores,
                      fit_platt, apply_platt, threshold_fpr_cap, row_metrics, write_json)
from pumpguard.features import FEATURE_NAMES

warnings.filterwarnings("ignore")


def evaluate(raw, y, dev, judged, folds):
    fit = dev & judged
    cal = fit_platt(raw[fit], y[fit])
    p = apply_platt(raw[dev], cal)
    thr = threshold_fpr_cap(y[fit], apply_platt(raw[fit], cal), FPR_CAP["standard"])
    m = row_metrics(y[dev], p, judged[dev], thr)
    pf = np.full(len(y), np.nan); pf[dev] = p
    fold = [row_metrics(y[te], pf[te], judged[te], thr)["f1"] for _, te in folds]
    return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "ap": m["ap_judged"], "fp": m["fp"],
            "fold_f1_min": float(min(fold)), "fold_f1_max": float(max(fold)), "fold_f1_sd": float(np.std(fold, ddof=1))}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="../evidence")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    d, _ = load_dataset(args.data_dir)
    X, history = make_features(d)
    y = d.label.to_numpy(); judged = history >= MIN_HISTORY
    dev = d.split.eq("development").to_numpy()
    folds = blocked_folds(d, dev)
    every = np.ones(len(d), bool)
    result = {"scope": "development split only (blocked 5-fold OOF); test untouched; nothing here changes the model"}

    vib_means = [FEATURE_NAMES.index("AI0_Vibration_mean_abs"), FEATURE_NAMES.index("AI1_Vibration_mean_abs")]
    fam = {}
    for label, idx in [("centered_11 (선정)", CENTERED), ("all_27_including_level", LEVEL27), ("vibration_window_means_only_2", vib_means)]:
        fam[label] = evaluate(oof_scores("hist_gb", X[:, idx], y, folds, judged), y, dev, judged, folds)
    result["feature_family_hist_gb"] = fam

    imb = {}
    for name in ("hist_gb", "extra_trees", "random_forest", "logistic"):
        for cw in ("balanced", None):
            imb[f"{name} / {'balanced' if cw else 'none'}"] = evaluate(oof_scores(name, X[:, CENTERED], y, folds, judged, class_weight=cw), y, dev, judged, folds)
    result["imbalance_weighting"] = imb

    result["training_rows_hist_gb"] = {
        "judged_only (선정)": fam["centered_11 (선정)"],
        "all_rows_including_warmup": evaluate(oof_scores("hist_gb", X[:, CENTERED], y, folds, every), y, dev, judged, folds)}

    sens = {"default (선정)": fam["centered_11 (선정)"]}
    for key, values in [("learning_rate", [.025, .1]), ("max_iter", [100, 400]), ("max_leaf_nodes", [7, 31]),
                        ("min_samples_leaf", [20, 80]), ("l2_regularization", [0.0, 5.0])]:
        for v in values:
            sens[f"{key}={v}"] = evaluate(oof_scores("hist_gb", X[:, CENTERED], y, folds, judged, params={key: v}), y, dev, judged, folds)
    result["hist_gb_sensitivity_one_at_a_time"] = sens

    raw = oof_scores("hist_gb", X[:, CENTERED], y, folds, judged)
    cal = {"none (sigmoid of balanced logit)": np.full(len(y), np.nan), "platt (선정)": np.full(len(y), np.nan), "isotonic": np.full(len(y), np.nan)}
    for _, te in folds:
        fit = dev & ~te & judged; ev = te & judged
        cal["none (sigmoid of balanced logit)"][ev] = 1 / (1 + np.exp(-raw[ev]))
        cal["platt (선정)"][ev] = apply_platt(raw[ev], fit_platt(raw[fit], y[fit]))
        cal["isotonic"][ev] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[fit], y[fit]).predict(raw[ev])
    m = dev & judged
    result["calibration_cross_fitted"] = {k: {"brier": float(brier_score_loss(y[m], v[m])), "ece_10bins": ece(y[m], v[m]),
                                              "mean_predicted": float(np.mean(v[m])), "observed_rate": float(np.mean(y[m]))}
                                          for k, v in cal.items()}
    write_json(out / "ablations.json", result)
    for section, table in result.items():
        if isinstance(table, dict):
            print(section)
            for k, v in table.items():
                print("  ", k, {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
