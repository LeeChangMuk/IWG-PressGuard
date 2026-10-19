"""SHAP explanation of the *shipped* v1.2 model (pumpguard/model.json, gradient-boosted trees).

Usage: python shap_analysis.py --data-dir data --output-dir ../evidence

- SHAP values are computed from the exported JSON trees (cover = training sample count per node)
  and cross-checked against shap.TreeExplainer on the deterministic sklearn refit (must agree to 1e-9).
- Values are in log-odds of the boosted model before Platt calibration (monotonic, rankings unchanged).
- Pairwise SHAP interaction values quantify which feature pairs act jointly.
- Four test cases are chosen by fixed rules (median score among rule matches), not by hand.
Nothing here changes the model, thresholds or test results.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from protocol import CENTERED, MIN_HISTORY, KOREAN, load_dataset, make_features, fit_candidate, apply_platt, write_json, korean_font
from pumpguard.features import FEATURE_NAMES
from pumpguard.inference import load_model, raw_score, LOWER_VIBRATION_STILL_STD

warnings.filterwarnings("ignore")


def json_tree_model(model):
    p = model["predictor"]
    if p["kind"] != "boosting":
        raise ValueError("shap_analysis.py v1.2 explains the boosting predictor")
    trees = []
    for t in p["trees"]:
        left = np.array(t["left"]); right = np.array(t["right"])
        trees.append({"children_left": left, "children_right": right, "children_default": left.copy(),
                      "features": np.array(t["feature"]), "thresholds": np.array(t["threshold"], dtype=np.float64),
                      "values": np.array(t["value"], dtype=np.float64).reshape(-1, 1),
                      "node_sample_weight": np.array(t["cover"], dtype=np.float64)})
    return {"trees": trees, "base_offset": float(p["baseline"]), "tree_output": "raw_value", "objective": "squared_error",
            "input_dtype": np.float64, "internal_dtype": np.float64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="../evidence")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "pumpguard" / "model.json"))
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    korean_font()
    model = load_model(args.model)
    idx = model["feature_indices"]; names = [FEATURE_NAMES[i] for i in idx]; ko = [KOREAN[n] for n in names]
    d, _ = load_dataset(args.data_dir)
    X, history = make_features(d)
    y = d.label.to_numpy(); judged = history >= MIN_HISTORY
    dev = d.split.eq("development").to_numpy(); test = ~dev
    sel = test & judged
    Xt = X[sel][:, idx]; yt = y[sel]

    ours = shap.TreeExplainer(json_tree_model(model))
    values = ours.shap_values(Xt, check_additivity=False); expected = float(np.ravel(ours.expected_value)[0])
    raw = np.array([raw_score(r.tolist(), model) for r in X[sel]])
    additivity = float(np.max(np.abs(values.sum(axis=1) + expected - raw)))
    sk = fit_candidate(model["selected_candidate"], X[dev & judged][:, CENTERED], y[dev & judged])
    if float(np.max(np.abs(sk.raw(Xt) - raw))) > 1e-9:
        raise AssertionError("sklearn refit differs from shipped JSON model")
    sk_explainer = shap.TreeExplainer(sk.model)
    reference = sk_explainer.shap_values(Xt)
    reference = reference[..., 1] if np.ndim(reference) == 3 else reference
    json_vs_sklearn = float(np.max(np.abs(reference - values)))
    if additivity > 1e-6 or json_vs_sklearn > 1e-6:
        raise AssertionError(f"SHAP check failed: additivity {additivity}, json-vs-sklearn {json_vs_sklearn}")
    inter = sk_explainer.shap_interaction_values(Xt)
    if inter.ndim == 4:
        inter = inter[..., 1]
    mean_abs = np.abs(values).mean(axis=0)
    mi = np.abs(inter).mean(axis=0)
    pairs = sorted([(mi[i, j] * 2, names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))], reverse=True)
    main_share = float(np.trace(mi) / mi.sum())

    calibrated = apply_platt(raw, model["calibration"]); thr = model["threshold"]
    td = d.loc[sel].reset_index(drop=True); h = history[sel]
    cur = X[sel][:, FEATURE_NAMES.index("AI2_Current_rms")]
    hi_cut = float(np.quantile(X[dev & judged & (y == 0), FEATURE_NAMES.index("AI2_Current_rms")], 2 / 3))
    full = h >= 10

    def median_pick(mask):
        ix = np.flatnonzero(mask)
        if not len(ix):
            return None
        return int(ix[np.argsort(calibrated[ix])][len(ix) // 2])
    rules = {"A_normal_high_load_quiet": ("정상·고부하·경보 없음", "label 0, full window, current RMS >= dev-normal 2/3 quantile, below standard threshold",
                                          full & (yt == 0) & (cur >= hi_cut) & (calibrated < thr)),
             "B_true_detection": ("이상·탐지", "label 1, full window, at/above standard threshold", full & (yt == 1) & (calibrated >= thr)),
             "C_missed_anomaly": ("이상·미탐", "label 1, full window, below standard threshold", full & (yt == 1) & (calibrated < thr)),
             "D_false_alarm": ("정상·오경보", "label 0, full window, at/above standard threshold", full & (yt == 0) & (calibrated >= thr))}
    cases = []
    for key, (title, rule, mask) in rules.items():
        i = median_pick(mask)
        if i is None:
            continue
        contrib = sorted(zip(names, values[i], Xt[i]), key=lambda t: -abs(t[1]))
        cases.append({"case": key, "title": title, "rule": rule, "candidates": int(mask.sum()), "segment_id": td.segment_id[i],
                      "timestamp": td.TimeStamp[i], "original_csv_line": int(td.source_row[i]) + 2, "label": int(yt[i]),
                      "probability": float(calibrated[i]), "current_rms": float(cur[i]),
                      "contributions": [{"feature": n, "feature_ko": KOREAN[n], "shap": float(v), "value": float(x)} for n, v, x in contrib]})

    # figures --------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.15]})
    o = np.argsort(mean_abs)
    axes[0].barh([ko[i] for i in o], mean_abs[o], color="#1f7a8c")
    axes[0].set(title="시험 판정 행 평균 |SHAP| (로그오즈, 배포 JSON 트리)", xlabel="평균 |SHAP|")
    im = axes[1].imshow(mi, cmap="Purples")
    axes[1].set_xticks(range(len(ko)), ko, rotation=60, ha="right", fontsize=8); axes[1].set_yticks(range(len(ko)), ko, fontsize=8)
    for i in range(len(ko)):
        for j in range(len(ko)):
            if mi[i, j] >= np.sort(mi.ravel())[-14]:
                axes[1].text(j, i, f"{mi[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mi[i, j] > mi.max() / 2 else "black")
    axes[1].set_title(f"SHAP 상호작용 평균 |값| (대각 = 단독 효과, 단독 비중 {main_share:.0%})")
    fig.colorbar(im, ax=axes[1], fraction=.04)
    fig.tight_layout(); fig.savefig(out / "shap_global.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (_, a, b) in zip(axes, pairs[:2]):
        ia, ib = names.index(a), names.index(b)
        sc = ax.scatter(Xt[:, ia], values[:, ia], c=Xt[:, ib], s=6, cmap="coolwarm", alpha=.7,
                        vmin=np.quantile(Xt[:, ib], .02), vmax=np.quantile(Xt[:, ib], .98))
        ax.scatter(Xt[yt == 1, ia], values[yt == 1, ia], s=14, facecolor="none", edgecolor="#222", lw=.5, label="실제 이상")
        ax.axhline(0, color="#555", lw=.8); ax.set_xscale("symlog", linthresh=max(np.quantile(Xt[:, ia], .05), 1e-3))
        ax.set(xlabel=KOREAN[a], ylabel=f"{KOREAN[a]} SHAP", title=f"{KOREAN[a]} 효과는 {KOREAN[b]}에 따라 달라짐")
        fig.colorbar(sc, ax=ax, label=KOREAN[b]); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "shap_dependence.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    for ax, c in zip(axes.ravel(), cases):
        top = c["contributions"][:7][::-1]
        ax.barh([t["feature_ko"] for t in top], [t["shap"] for t in top], color=["#c0392b" if t["shap"] > 0 else "#1f7a8c" for t in top])
        ax.axvline(0, color="#555", lw=.8)
        ax.set_title(f"{c['title']} · 이상확률 {c['probability']:.3f} · 전류RMS {c['current_rms']:.0f} · {c['segment_id']}", fontsize=10)
        ax.set_xlabel("SHAP (로그오즈, +는 이상 방향)")
    fig.suptitle(f"배포 모델 SHAP 사례 (기준값 {expected:.2f}, 시험 구간, 사례는 고정 규칙의 중앙값)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "shap_cases.png", dpi=150); plt.close(fig)

    # correlation reversal (descriptive, all data)
    ci = FEATURE_NAMES.index("vibration_correlation")
    seg = d.groupby("segment_id").filter(lambda g: len(g) >= 10).groupby("segment_id").apply(
        lambda g: pd.Series({"label": g.label.iloc[0], "r": np.corrcoef(g.AI0_Vibration, g.AI1_Vibration)[0, 1]}), include_groups=False)
    file_r = {int(k): float(np.corrcoef(g.AI0_Vibration, g.AI1_Vibration)[0, 1]) for k, g in d.groupby("label")}
    win = {int(l): float(np.median(X[(y == l) & (history >= 10), ci])) for l in (0, 1)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(-1, 1, 41)
    for l, col, t in [(0, "#1f7a8c", "정상"), (1, "#c0392b", "이상")]:
        axes[0].hist(X[(y == l) & (history >= 10), ci], bins=bins, density=True, alpha=.55, color=col, label=t)
        axes[1].hist(seg.loc[seg.label.eq(l), "r"], bins=bins, density=True, alpha=.55, color=col, label=t)
    axes[0].set(title="10표본 창 진동 상관(전체 기간)", xlabel="상관계수", ylabel="밀도"); axes[0].legend()
    axes[1].set(title=f"연속구간 상관 · 파일 전체 정상 {file_r[0]:+.3f} / 이상 {file_r[1]:+.3f}", xlabel="상관계수"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "vibration_correlation.png", dpi=150); plt.close(fig)

    # conditional effects quoted in Chapter 3 (test judged rows)
    col = lambda n: names.index(n)
    cs, cr = Xt[:, col("AI2_Current_std")], Xt[:, col("AI2_Current_residual_abs")]
    flat = (cs <= np.quantile(cs, 1 / 3)) & (cr <= np.quantile(cr, 1 / 3))
    lower = Xt[:, col("AI1_Vibration_std")] <= LOWER_VIBRATION_STILL_STD
    corr_neg = Xt[:, col("vibration_correlation")] < 0
    conditional = {
        "current_std_shap_mean_flat_current": float(values[flat, col("AI2_Current_std")].mean()),
        "current_std_shap_mean_top_third_current_std": float(values[cs >= np.quantile(cs, 2 / 3), col("AI2_Current_std")].mean()),
        "flat_current_definition": "current std and current residual both in their lowest third (test judged rows)",
        "correlation_shap_mean_negative_correlation": float(values[corr_neg, col("vibration_correlation")].mean()),
        "correlation_shap_mean_nonnegative_correlation": float(values[~corr_neg, col("vibration_correlation")].mean()),
        "lower_vibration_still_rows": int(lower.sum()),
        "lower_vibration_still_ai1_std_shap_quartiles": np.quantile(values[lower, col("AI1_Vibration_std")], [.25, .5, .75]).tolist()}

    summary = {"model_id": model["model_id"], "explained_output": "log-odds of the boosted model before Platt calibration",
               "method": "path-dependent TreeSHAP on the exported JSON trees; cross-checked with shap.TreeExplainer on the sklearn refit",
               "shap_version": shap.__version__, "rows": int(len(Xt)), "expected_value": expected,
               "max_additivity_error": additivity, "max_json_vs_sklearn_difference": json_vs_sklearn,
               "mean_abs_shap": {n: float(v) for n, v in sorted(zip(names, mean_abs), key=lambda t: -t[1])},
               "interaction_main_effect_share": main_share,
               "top_interactions": [{"pair": [a, b], "pair_ko": [KOREAN[a], KOREAN[b]], "mean_abs_interaction_x2": float(v)} for v, a, b in pairs[:6]],
               "vibration_correlation": {"file_level": file_r, "segment_median": {int(k): float(s.median()) for k, s in seg.groupby("label").r},
                                         "segment_share_negative": {int(k): float((s < 0).mean()) for k, s in seg.groupby("label").r},
                                         "segments": {int(k): int(len(s)) for k, s in seg.groupby("label").r}, "window_median_full": win},
               "conditional_effects": conditional,
               "cases": cases, "interpretation_limit": "Explains this model on this dataset; not proof of a physical fault mechanism."}
    write_json(out / "shap_summary.json", summary)
    pd.DataFrame([{"case": c["case"], "title": c["title"], **t, "probability": c["probability"], "segment_id": c["segment_id"]}
                  for c in cases for t in c["contributions"]]).to_csv(out / "shap_cases.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"additivity": additivity, "json_vs_sklearn": json_vs_sklearn, "mean_abs": summary["mean_abs_shap"],
                      "main_share": main_share, "pairs": summary["top_interactions"][:4],
                      "cases": [(c["case"], round(c["probability"], 3), [(t["feature"], round(t["shap"], 2)) for t in c["contributions"][:3]]) for c in cases]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
