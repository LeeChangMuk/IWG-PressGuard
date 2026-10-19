"""Diagnostics of the frozen v1.2 model for Chapters 2-4. Never refits the deployed model's choice,
never changes a threshold; the sklearn refit below is deterministic and is checked to reproduce the
shipped JSON scores exactly before it is used for fast permutation tests.

Usage: python analysis.py --data-dir data --output-dir ../evidence   (after train.py)
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
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from protocol import (CENTERED, MIN_HISTORY, FPR_CAP, KOREAN, load_dataset, make_features, blocked_folds, oof_scores,
                      fit_candidate, fit_platt, apply_platt, threshold_fpr_cap, row_metrics, write_json, korean_font)
from pumpguard.features import CHANNELS, FEATURE_NAMES
from pumpguard.inference import load_model, LOWER_VIBRATION_STILL_STD

warnings.filterwarnings("ignore")
MODES = ("unmanned", "standard", "screening")
MODE_KO = {"unmanned": "무인 감시", "standard": "표준 운전", "screening": "점검 선별"}
GROUPS = {
    "진동 진폭(AI0·AI1 표준편차)": ["AI0_Vibration_std", "AI1_Vibration_std"],
    "전류 진폭(전류 표준편차)": ["AI2_Current_std"],
    "순간편차 3개": ["AI0_Vibration_residual_abs", "AI1_Vibration_residual_abs", "AI2_Current_residual_abs"],
    "변화량 3개": ["AI0_Vibration_diff_abs", "AI1_Vibration_diff_abs", "AI2_Current_diff_abs"],
    "진동 채널 관계(상관·SD 로그비)": ["vibration_correlation", "vibration_std_log_ratio"],
}


def ece(y, p, edges):
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="../evidence")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    korean_font()
    model = load_model(out / "model.json")
    selected = model["selected_candidate"]; thr = {k: model["operating_modes"][k]["threshold"] for k in MODES}
    d, _ = load_dataset(args.data_dir)
    X, history = make_features(d)
    d["n_history"] = history
    y = d.label.to_numpy(); judged = history >= MIN_HISTORY
    dev = d.split.eq("development").to_numpy(); test = ~dev
    Xc = X[:, CENTERED]
    final = fit_candidate(selected, Xc[dev & judged], y[dev & judged])
    raw_test = final.raw(Xc[test]); p_test = apply_platt(raw_test, model["calibration"])
    # round_trip parsing: the standard threshold IS one observed OOF score, so the default fast float
    # parser (last-digit error) would turn the boundary row into a miss.
    saved = pd.read_csv(out / "predictions_test.csv", float_precision="round_trip")
    if not np.allclose(saved.score.to_numpy(), p_test, atol=1e-9):
        raise AssertionError("Refit does not reproduce the shipped model scores")
    oof = pd.read_csv(out / "oof_development.csv", float_precision="round_trip")
    p_dev = np.full(len(d), np.nan); p_dev[dev] = oof["oof_" + selected].to_numpy()
    oof_check = row_metrics(y[dev], p_dev[dev], judged[dev], thr["standard"])
    reported = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["development_oof_selected_modes"]["standard"]
    if (oof_check["tp"], oof_check["fp"]) != (reported["tp"], reported["fp"]):
        raise AssertionError("Development OOF decisions differ from train.py")
    ty, tj = y[test], judged[test]
    result = {"selected_model": selected, "thresholds": thr}

    # ------------------------------------------------------------ 1. calibration
    edges = [0, .02, .05, .1, .2, .4, .6, .8, 1.0]
    before = 1 / (1 + np.exp(-raw_test))
    bins = []
    for name, p in (("보정 전(균형가중 확률)", before), ("Platt 보정 후", p_test)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = tj & (p >= lo) & ((p < hi) if hi < 1 else (p <= 1))
            if m.any():
                bins.append({"calibration": name, "bin_low": lo, "bin_high": hi, "rows": int(m.sum()),
                             "mean_predicted": float(p[m].mean()), "observed_anomaly_rate": float(ty[m].mean())})
    pd.DataFrame(bins).to_csv(out / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    result["calibration_test_judged"] = {
        "brier_before": float(brier_score_loss(ty[tj], before[tj])), "brier_after": float(brier_score_loss(ty[tj], p_test[tj])),
        "ece_before": ece(ty[tj], before[tj], edges), "ece_after": ece(ty[tj], p_test[tj], edges),
        "mean_predicted_after": float(p_test[tj].mean()), "observed_rate": float(ty[tj].mean()),
        "note": "Calibration fitted on development OOF only. Probabilities refer to the in-dataset anomaly share."}

    # ------------------------------------------------------------ 2. importance
    rng = np.random.default_rng(20260928)
    base_ap = average_precision_score(ty[tj], p_test[tj]); A = Xc[test][tj]
    perm = []
    for j, fi in enumerate(CENTERED):
        drops = []
        for _ in range(30):
            Z = A.copy(); Z[:, j] = rng.permutation(Z[:, j])
            drops.append(base_ap - average_precision_score(ty[tj], apply_platt(final.raw(Z), model["calibration"])))
        perm.append({"feature": FEATURE_NAMES[fi], "feature_ko": KOREAN[FEATURE_NAMES[fi]], "ap_drop_mean": float(np.mean(drops)),
                     "ap_drop_p2.5": float(np.quantile(drops, .025)), "ap_drop_p97.5": float(np.quantile(drops, .975))})
    perm.sort(key=lambda r: -r["ap_drop_mean"])
    pd.DataFrame(perm).to_csv(out / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    result["permutation_importance_test"] = {"base_ap_judged": base_ap, "repeats": 30, "method": "global shuffle of one feature over judged test rows", "rows": perm}

    folds = blocked_folds(d, dev)

    def oof_eval(idx):
        raw = oof_scores(selected, X[:, idx], y, folds, judged)
        fit = dev & judged; cal = fit_platt(raw[fit], y[fit]); p = apply_platt(raw[dev], cal)
        t = threshold_fpr_cap(y[fit], apply_platt(raw[fit], cal), FPR_CAP["standard"])
        m = row_metrics(y[dev], p, judged[dev], t)
        return m["f1"], m["ap_judged"]
    full_f1, full_ap = oof_eval(CENTERED)
    groups = []
    for gname, members in GROUPS.items():
        keep = [i for i in CENTERED if FEATURE_NAMES[i] not in members]
        f1, ap = oof_eval(keep)
        groups.append({"group": gname, "features_removed": len(members), "oof_f1_without": f1, "oof_ap_without": ap,
                       "f1_drop": full_f1 - f1, "ap_drop": full_ap - ap})
    groups.sort(key=lambda r: -r["ap_drop"])
    pd.DataFrame(groups).to_csv(out / "group_importance.csv", index=False, encoding="utf-8-sig")
    result["group_drop_column_development"] = {"full_oof_f1": full_f1, "full_oof_ap": full_ap, "rows": groups,
                                               "method": "refit the selected model without a feature group; development blocked OOF"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    pp = perm[::-1]
    axes[0].barh([r["feature_ko"] for r in pp], [r["ap_drop_mean"] for r in pp], color="#1f7a8c",
                 xerr=[[r["ap_drop_mean"] - r["ap_drop_p2.5"] for r in pp], [r["ap_drop_p97.5"] - r["ap_drop_mean"] for r in pp]], ecolor="#333")
    axes[0].set(title="시험: 특징 하나를 섞었을 때 AP 감소 (30회 반복, 막대=평균, 선=95%)", xlabel="AP 감소")
    gg = groups[::-1]
    axes[1].barh([r["group"] for r in gg], [r["ap_drop"] for r in gg], color="#c0392b")
    for i, r in enumerate(gg):
        axes[1].text(max(r["ap_drop"], 0) + .002, i, f"F1 {r['f1_drop']:+.3f}", va="center", fontsize=8)
    axes[1].axvline(0, color="#555", lw=.8)
    axes[1].set(title="개발 OOF: 특징 묶음을 빼고 다시 학습했을 때 AP 감소", xlabel="AP 감소 (전체 AP {:.3f})".format(full_ap))
    fig.tight_layout(); fig.savefig(out / "importance.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------ 3. where FN / FP concentrate
    cur = X[:, FEATURE_NAMES.index("AI2_Current_rms")]; vib = X[:, FEATURE_NAMES.index("AI0_Vibration_std")]
    corr = X[:, FEATURE_NAMES.index("vibration_correlation")]
    ref = dev & judged & (y == 0)
    load_cut = np.quantile(cur[ref], [1 / 3, 2 / 3]); vib_cut = float(np.quantile(vib[ref], .95))
    load = np.digitize(cur, load_cut); high_vib = vib > vib_cut
    score = np.where(dev, p_dev, np.nan); score[test] = p_test
    pred = judged & (score >= thr["standard"])
    pool = judged.copy()                                          # every judged row has an out-of-sample score
    cells = []
    for scope, mask in (("개발 OOF + 시험(표본 외 전체)", pool), ("시험만", pool & test)):
        for li, lname in enumerate(["저부하", "중부하", "고부하"]):
            for hv, vname in ((False, "통상 진동"), (True, "높은 진동")):
                m = mask & (load == li) & (high_vib == hv)
                n0 = int(np.sum(m & (y == 0))); n1 = int(np.sum(m & (y == 1)))
                fp = int(np.sum(m & (y == 0) & pred)); fn = int(np.sum(m & (y == 1) & ~pred))
                cells.append({"scope": scope, "load": lname, "vibration": vname, "normal_rows": n0, "fp": fp,
                              "fp_rate": fp / n0 if n0 else None, "anomaly_rows": n1, "fn": fn, "fn_rate": fn / n1 if n1 else None})
    cdf = pd.DataFrame(cells); cdf.to_csv(out / "error_conditions.csv", index=False, encoding="utf-8-sig")
    tot = {s: {"fp": int(g.fp.sum()), "fn": int(g.fn.sum()), "normal": int(g.normal_rows.sum()), "anomaly": int(g.anomaly_rows.sum())}
           for s, g in cdf.groupby("scope", sort=False)}
    extra = {}
    for scope, mask in (("pooled", pool), ("test", pool & test)):
        extra[scope] = {
            "by_correlation_sign": {nm: {"normal_rows": int(np.sum(mask & c & (y == 0))), "fp": int(np.sum(mask & c & (y == 0) & pred)),
                                         "anomaly_rows": int(np.sum(mask & c & (y == 1))), "fn": int(np.sum(mask & c & (y == 1) & ~pred))}
                                    for nm, c in (("corr<0", corr < 0), ("corr>=0", corr >= 0))},
            "by_history": {nm: {"normal_rows": int(np.sum(mask & c & (y == 0))), "fp": int(np.sum(mask & c & (y == 0) & pred)),
                                "anomaly_rows": int(np.sum(mask & c & (y == 1))), "fn": int(np.sum(mask & c & (y == 1) & ~pred))}
                           for nm, c in (("3-4", (history >= 3) & (history <= 4)), ("5-9", (history >= 5) & (history <= 9)), ("10", history == 10))}}
    # automatic slice discovery: depth-2 trees on interpretable condition variables
    cond = np.column_stack([cur, vib, X[:, FEATURE_NAMES.index("AI1_Vibration_std")], corr, history])
    cname = ["전류 RMS(부하)", "AI0 진동 SD", "AI1 진동 SD", "진동 상관", "이력 표본수"]

    def slices(target_mask, err, min_leaf):
        t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=min_leaf, random_state=0).fit(cond[target_mask], err[target_mask])
        tr = t.tree_; leaves = t.apply(cond[target_mask]); out_rows = []
        def path(node, rule):
            if tr.children_left[node] == -1:
                m = leaves == node
                out_rows.append({"rule": " 그리고 ".join(rule) or "전체", "rows": int(m.sum()), "errors": int(err[target_mask][m].sum()),
                                 "error_rate": float(err[target_mask][m].mean())}); return
            f, v = cname[tr.feature[node]], tr.threshold[node]
            path(tr.children_left[node], rule + [f"{f} ≤ {v:.3g}"]); path(tr.children_right[node], rule + [f"{f} > {v:.3g}"])
        path(0, [])
        return sorted(out_rows, key=lambda r: -r["error_rate"])
    fp_err = (pred & (y == 0)).astype(int); fn_err = (~pred & (y == 1)).astype(int)
    result["error_conditions"] = {"load_cuts_current_rms": load_cut.tolist(), "high_vibration_cut_ai0_std": vib_cut,
                                  "definition": "Cuts from development normal judged rows. Current RMS is a load proxy; no product/speed labels exist.",
                                  "totals": tot, "other_splits": extra,
                                  "fp_slices_normal_rows": slices(pool & (y == 0), fp_err, 300),
                                  "fn_slices_anomaly_rows": slices(pool & (y == 1), fn_err, 25),
                                  "overall_fp_rate_pooled": float(fp_err[pool & (y == 0)].mean()),
                                  "overall_fn_rate_pooled": float(fn_err[pool & (y == 1)].mean())}
    # the deployed evidence rule (pumpguard/inference.py) for the main false-alarm condition
    still = X[:, FEATURE_NAMES.index("AI1_Vibration_std")] <= LOWER_VIBRATION_STILL_STD
    share = lambda m: float(np.mean(still[m]))
    result["lower_vibration_still"] = {
        "rule": f"AI1_Vibration_std <= {LOWER_VIBRATION_STILL_STD} (judged rows)",
        "share_anomalous_rows": share(judged & (y == 1)), "share_normal_rows": share(judged & (y == 0)),
        "share_normal_rows_development": share(judged & (y == 0) & dev), "share_normal_rows_test": share(judged & (y == 0) & test),
        "pooled_fp_inside": int(np.sum(still & fp_err.astype(bool) & pool)), "pooled_fp_total": int(fp_err[pool].sum()),
        "pooled_fn_inside": int(np.sum(still & fn_err.astype(bool) & pool)), "pooled_fn_total": int(fn_err[pool].sum())}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    sub = cdf[cdf.scope.str.startswith("개발")]
    for ax, (rate, cnt, tot_col, title, cmap) in zip(axes, [("fp_rate", "fp", "normal_rows", "정상 표본 중 오경보 비율(FP)", "Blues"),
                                                           ("fn_rate", "fn", "anomaly_rows", "이상 표본 중 미탐 비율(FN)", "Reds")]):
        grid = np.full((2, 3), np.nan); txt = [["" for _ in range(3)] for _ in range(2)]
        for _, r in sub.iterrows():
            i = 0 if r.vibration == "통상 진동" else 1; j = ["저부하", "중부하", "고부하"].index(r.load)
            if r[tot_col] >= 30:
                grid[i, j] = r[rate]; txt[i][j] = f"{r[rate]:.1%}\n({r[cnt]}/{r[tot_col]:,})"
            else:
                txt[i][j] = f"표본 부족\n({r[cnt]}/{r[tot_col]:,})"
        ax.imshow(np.nan_to_num(grid, nan=0), cmap=cmap, vmin=0, vmax=max(np.nanmax(grid), .01))
        for i in range(2):
            for j in range(3):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), ["저부하", "중부하", "고부하"]); ax.set_yticks(range(2), ["통상 진동", "높은 진동"])
        ax.set_title(title + " — 표본 외 전체", fontsize=10)
    fig.suptitle("운전 조건(전류 RMS 3분위 × AI0 진동 SD 95% 기준)별 오류 집중도, 표준 모드", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "error_conditions.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------ 4. inspection priority (screening ranking)
    def ranking(mask, scores):
        g = pd.DataFrame({"segment_id": d.segment_id.to_numpy()[mask], "label": y[mask], "score": scores[mask], "judged": judged[mask]})
        g = g[g.judged]
        t = g.groupby("segment_id", sort=False).agg(label=("label", "first"), flag_ratio=("score", lambda s: float(np.mean(s >= thr["screening"]))),
                                                    mean_score=("score", "mean"), rows=("score", "size")).reset_index()
        t = t.sort_values(["flag_ratio", "mean_score"], ascending=False).reset_index(drop=True); t["rank"] = np.arange(1, len(t) + 1)
        ranks = t.loc[t.label.eq(1), "rank"].tolist()
        return t, {"segments": len(t), "anomalous_ranks": ranks, "inspect_top_n_to_cover_all": int(max(ranks)),
                   "inspect_share_to_cover_all": max(ranks) / len(t)}
    ttab, trank = ranking(test, score); dtab, drank = ranking(dev, score)
    ttab.to_csv(out / "screening_segment_ranking_test.csv", index=False, encoding="utf-8-sig")
    result["inspection_ranking"] = {"rule": "segments ordered by share of judged samples >= screening threshold; tie-break mean probability",
                                    "test": trank, "development_oof": drank,
                                    "test_gain_curve": [{"inspected_share": (i + 1) / len(ttab), "anomalous_found": int(ttab.label.iloc[:i + 1].sum())} for i in range(len(ttab))]}

    # ------------------------------------------------------------ 5. decision-cost view of the thresholds
    prior = float(np.mean(y[dev & judged]))
    result["decision_cost_view"] = {
        "rule": "With calibrated probability p, alarm when p >= t is optimal if C_FN/C_FP = (1-t)/t",
        "implied_cost_ratio_fn_over_fp": {k: (1 - v) / v for k, v in thr.items()},
        "dataset_prior": prior,
        "prior_shift_formula": "odds_field = odds_dataset * [pi_f/(1-pi_f)] / [pi_d/(1-pi_d)]",
        "example_field_prior_0.5pct": {k: float((v / (1 - v)) * ((0.005 / 0.995) / (prior / (1 - prior))) / (1 + (v / (1 - v)) * ((0.005 / 0.995) / (prior / (1 - prior))))) for k, v in thr.items()}}

    # ------------------------------------------------------------ 6. traceable error examples (test)
    td = d.loc[test].reset_index(drop=True); sv = saved
    fp = sv[(sv.label == 0) & (sv.decision_standard == 1)].sort_values("score", ascending=False)
    fn = sv[(sv.label == 1) & (sv.judged == 1) & (sv.decision_standard == 0)].sort_values("score")
    examples, manifest = [], []
    for kind, errors in (("FALSE_POSITIVE", fp), ("FALSE_NEGATIVE", fn)):
        for _, e in errors.drop_duplicates("segment_id").head(3).iterrows():
            eid = f"E{len(manifest) + 1:02d}"
            g = td[td.segment_id.eq(e.segment_id)]
            target = g.index[(g.source_row + 2).eq(e.original_csv_line)].item()
            for i, r in g[g.index <= target].tail(10).iterrows():
                examples.append({"example_id": eid, "error_type": kind, "is_target": i == target, "segment_id": r.segment_id,
                                 "source_file": r.source_file, "original_csv_line": int(r.source_row) + 2, "TimeStamp": r.TimeStamp,
                                 **{c: r[c] for c in CHANNELS}, "label": int(r.label), "score": float(sv.score.iloc[i]),
                                 "status": sv.status_standard.iloc[i]})
            manifest.append({"example_id": eid, "error_type": kind, "segment_id": e.segment_id, "source_file": e.source_file,
                             "original_csv_line": int(e.original_csv_line), "score": float(e.score), "threshold": thr["standard"]})
    pd.DataFrame(examples).to_csv(out / "error_examples.csv", index=False, encoding="utf-8-sig")
    write_json(out / "error_examples_manifest.json", manifest)
    write_json(out / "analysis.json", result)

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8")); ts = m["test"]["rows"]["standard"]; tp = m["test"]["persistent"]["standard"]
    card = f"""# 프레스가드 라이트 모델 카드 (v1.2.0)

- 모델: {selected} (히스토그램 그래디언트 부스팅 200회, 리프 15, 학습률 0.05, L2 1.0, 균형 가중치) + Platt 보정. 팀: IWG.
- 용도: KAMP 소성가공 데이터의 정상·이상 상태 분류와 보정된 이상확률 제공, 현장 CSV 섀도 관찰·점검 우선순위·작업자 판단 보조.
- 비대상: 설비 제어·안전정지, 고장 원인 확정, 현장 고장확률 보증, 고장 전 리드타임·잔여수명 추정.
- 입력: 0.1초 간격 TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current (단위·AI0/AI1 위치는 현장 확인 필요).
- 특징: 과거~현재 최대 10표본 창의 오프셋 불변 특징 11개(채널별 표준편차·순간편차·변화량, 진동 상관, 진동 SD 로그비).
- 판정: 이력 3표본 이상 행만 판정·투표. 최근 판정 3개 중 2개 초과 시 경보. 공백 0.15초 초과 시 초기화.
- 검증: 파일별 시간순 마지막 20% 구간 = 시험(1회). 앞 80%는 시간 블록 5-겹 교차검증(퍼지 1구간)으로 선정·보정·임계값 결정.
- 임계값: 무인 {thr['unmanned']:.4f} / 표준 {thr['standard']:.4f} / 선별 {thr['screening']:.4f} (개발 OOF 규칙).
- 시험(표준): 표본 F1 {ts['f1']:.3f}, 정밀도 {ts['precision']:.3f}, 재현율 {ts['recall']:.3f}, AP {ts['ap_judged']:.3f}; 지속경보 F1 {tp['f1']:.3f}, 이상 구간 {tp['anomalous_segments_detected']}/{tp['anomalous_segments_total']}.
- 확률: 개발 OOF로 보정한 데이터셋 내 이상확률(이상 비율 약 2.9% 기준). 현장 비율이 다르면 사전확률 보정 필요.
- 한계: 라벨=파일=날짜, 이상 기록 1개(2.8분), 고장 시작·정비 이력 없음, 현장 실증 미실시.
- 재현: python reproduce.py (원본 해시 확인 후 전 과정 자동 실행, 모델 바이트 동일성 확인).
"""
    (out / "MODEL_CARD.md").write_bytes(card.encode("utf-8"))
    print(json.dumps({"calibration": result["calibration_test_judged"], "group": [(g["group"], round(g["ap_drop"], 4), round(g["f1_drop"], 4)) for g in groups],
                      "perm_top": [(p["feature"], round(p["ap_drop_mean"], 4)) for p in perm[:5]], "ranking": {"test": trank, "dev": drank},
                      "fp_slices": result["error_conditions"]["fp_slices_normal_rows"][:2], "fn_slices": result["error_conditions"]["fn_slices_anomaly_rows"][:2]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
