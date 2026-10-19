"""PressGuard v1.2 fixed analysis protocol: data, splits, candidates, exporters, metrics.

Design rules (fixed before any v1.2 test evaluation):
- Test split = the last 20% of time-ordered segments in each source file (unchanged since v1.0).
- Development split = the first 80%. Every choice (model, calibration, thresholds) uses
  out-of-fold (OOF) predictions from a blocked, purged 5-fold CV inside development.
- Features = 11 offset-invariant ("centered") causal features. Level features are excluded
  because labels coincide with file/date, so a per-file offset cannot be told apart from a fault.
- A row is judged only when its causal history has >= 3 samples (n=1 gives an all-zero vector,
  n=2 gives a correlation of exactly +/-1). Warm-up rows never vote, never alarm and count as
  "no alarm" in every metric. Models are trained on judged rows only.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from pumpguard.features import CHANNELS, FEATURE_NAMES, CausalFeatureExtractor

SEED = 20260928
TEST_SHARE = 0.20
DEV_FOLDS = 5
PURGE_SEGMENTS = 1
MIN_HISTORY = 3
FPR_CAP = {"unmanned": 0.001, "standard": 0.01}
SCREENING_RECALL = 0.95  # must exceed the standard mode's OOF recall (~0.90) to be a distinct, more sensitive list
CENTERED = [i for i, n in enumerate(FEATURE_NAMES)
            if n.endswith(("_std", "_residual_abs", "_diff_abs")) or n in ("vibration_correlation", "vibration_std_log_ratio")]
LEVEL27 = list(range(len(FEATURE_NAMES)))
STD3 = [FEATURE_NAMES.index(c + "_std") for c in CHANNELS]

KOREAN = {
    "AI0_Vibration_std": "AI0 진동 표준편차", "AI0_Vibration_residual_abs": "AI0 진동 순간편차", "AI0_Vibration_diff_abs": "AI0 진동 변화량",
    "AI1_Vibration_std": "AI1 진동 표준편차", "AI1_Vibration_residual_abs": "AI1 진동 순간편차", "AI1_Vibration_diff_abs": "AI1 진동 변화량",
    "AI2_Current_std": "전류 표준편차", "AI2_Current_residual_abs": "전류 순간편차", "AI2_Current_diff_abs": "전류 변화량",
    "vibration_correlation": "AI0-AI1 진동 상관", "vibration_std_log_ratio": "AI0/AI1 진동 SD 로그비",
}

# Candidate list and hyper-parameters are fixed a priori (no tuning on 16 anomalous segments).
CANDIDATES = [
    {"name": "rule_std", "label": "단일센서 한계값(현장 관행형 기준선)", "family": "기준선", "eligible": True,
     "note": "채널별 창 표준편차의 정상 기준 강건 z점수 최댓값(정상 데이터만 사용)"},
    {"name": "isolation_forest", "label": "Isolation Forest(비지도 기준선)", "family": "기준선", "eligible": False,
     "note": "정상 데이터만 학습. 라벨을 쓰지 않는 현장 초기 단계용 참조 모델(선정 대상 아님)"},
    {"name": "logistic", "label": "로지스틱 회귀", "family": "선형", "eligible": True, "note": "표준화 + L2(C=0.3), 균형 가중치"},
    {"name": "decision_tree", "label": "결정트리(깊이 4)", "family": "트리", "eligible": True, "note": "규칙 해석형 단일 트리, 리프 최소 20"},
    {"name": "random_forest", "label": "랜덤포레스트", "family": "배깅 앙상블", "eligible": True, "note": "64그루, 깊이 8, sqrt 특징 무작위"},
    {"name": "extra_trees", "label": "ExtraTrees(v1.0 모델 구성)", "family": "배깅 앙상블", "eligible": True, "note": "64그루, 깊이 6, 무작위 분할점"},
    {"name": "hist_gb", "label": "히스토그램 그래디언트 부스팅", "family": "부스팅 앙상블", "eligible": True,
     "note": "200회, 학습률 0.05, 리프 15, L2 1.0, 조기종료 없음"},
    {"name": "soft_vote", "label": "이종 소프트보팅(로지스틱+ExtraTrees+부스팅)", "family": "이종 앙상블", "eligible": True,
     "note": "세 모델 확률 평균"},
]
SIMPLICITY = {c["name"]: i for i, c in enumerate(CANDIDATES)}
CANDIDATE_BY_NAME = {c["name"]: c for c in CANDIDATES}


def korean_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_json(path, obj):
    Path(path).write_bytes(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ----------------------------------------------------------------------------- data
def load_dataset(data_dir):
    frames, sources = [], []
    for name, label, prefix in [("press_data_normal.csv", 0, "N"), ("outlier_data.csv", 1, "A")]:
        found = list(Path(data_dir).rglob(name))
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {name}; found {len(found)}")
        path = found[0]
        d = pd.read_csv(path)
        if not all(c in d for c in ["TimeStamp", *CHANNELS, "Equipment_state"]):
            raise ValueError("CSV schema mismatch")
        if not d.Equipment_state.eq(label).all():
            raise ValueError("Unexpected source label")
        if not np.isfinite(d[list(CHANNELS)].values).all():
            raise ValueError("Nonfinite training input")
        d["source_row"] = np.arange(len(d))
        before = len(d)
        d = d.drop_duplicates(subset=["TimeStamp", *CHANNELS, "Equipment_state"], keep="first").copy()
        t = pd.to_datetime(d.TimeStamp, errors="raise")
        delta = t.diff().dt.total_seconds()
        if (delta <= 0).any():
            raise ValueError("Nonincreasing timestamp remains after exact deduplication")
        segments = (delta.isna() | (delta > 0.15)).cumsum()
        d["segment_id"] = [f"{prefix}{v:04d}" for v in segments]
        ids = list(d.segment_id.unique())
        n = len(ids)
        cut = int(n * (1 - TEST_SHARE))          # identical boundary to v1.0 (floor of 80%)
        dev_ids = ids[:cut]
        d["split"] = np.where(d.segment_id.isin(dev_ids), "development", "test")
        # contiguous development blocks for the blocked CV (block k = k-th fifth of dev segments)
        bounds = [int(round(len(dev_ids) * i / DEV_FOLDS)) for i in range(DEV_FOLDS + 1)]
        block = {sid: k for k in range(DEV_FOLDS) for sid in dev_ids[bounds[k]:bounds[k + 1]]}
        d["dev_block"] = d.segment_id.map(block).fillna(-1).astype(int)
        d["segment_order"] = d.segment_id.map({sid: i for i, sid in enumerate(ids)})
        d["source_file"] = name
        d["label"] = label
        sources.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "raw_rows": before,
                        "deduplicated_rows": len(d), "removed_duplicates": before - len(d), "segments": n,
                        "development_segments": len(dev_ids), "test_segments": n - len(dev_ids)})
        frames.append(d)
    return pd.concat(frames, ignore_index=True), sources


def make_features(d, transform=None):
    rows, history = [], []
    for _, g in d.groupby("segment_id", sort=False):
        extractor = CausalFeatureExtractor()
        for row in g.itertuples():
            values = [getattr(row, c) for c in CHANNELS]
            if transform:
                values = transform(values)
            result = extractor.update(values, row.TimeStamp)
            rows.append(result["features"])
            history.append(result["n_history"])
    return np.asarray(rows), np.asarray(history)


def blocked_folds(d, mask, k=DEV_FOLDS, purge=PURGE_SEGMENTS, block_column=None):
    """Contiguous time blocks of segments inside each file; neighbours of the held-out block are purged.

    Returns a list of (train_mask, test_mask) over all rows of d. Only rows in `mask` are used."""
    seg = d.segment_id.to_numpy()
    plan = {}
    for f, g in d[mask].groupby("source_file", sort=False):
        ids = list(g.sort_values("segment_order").segment_id.unique())
        if block_column is not None:
            blocks = g.drop_duplicates("segment_id").set_index("segment_id")[block_column].to_dict()
            assignment = [blocks[s] for s in ids]
        else:
            b = [int(round(len(ids) * i / k)) for i in range(k + 1)]
            assignment = [next(j for j in range(k) if b[j] <= i < b[j + 1]) for i in range(len(ids))]
        plan[f] = (ids, assignment)
    folds = []
    for fold in range(k):
        held, purged = set(), set()
        for ids, assignment in plan.values():
            positions = [i for i, a in enumerate(assignment) if a == fold]
            if not positions:
                continue
            held.update(ids[i] for i in positions)
            lo, hi = min(positions), max(positions)
            for j in range(1, purge + 1):
                if lo - j >= 0:
                    purged.add(ids[lo - j])
                if hi + j < len(ids):
                    purged.add(ids[hi + j])
        test_mask = mask & np.isin(seg, list(held))
        train_mask = mask & ~test_mask & ~np.isin(seg, list(purged))
        folds.append((train_mask, test_mask))
    return folds


# ----------------------------------------------------------------------------- candidates
class Fitted:
    """Wraps a fitted candidate. raw(X) returns the score that is Platt-calibrated (a logit for
    probabilistic models). proba(X) returns the uncalibrated class-1 probability where defined."""

    def __init__(self, name, model, raw, proba=None):
        self.name, self.model, self._raw, self._proba = name, model, raw, proba

    def raw(self, X):
        return self._raw(X)

    def proba(self, X):
        return self._proba(X) if self._proba else None


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_candidate(name, X, y, class_weight="balanced", params=None):
    params = params or {}
    if name == "rule_std":
        idx = [CENTERED.index(i) for i in STD3]
        normal = X[y == 0][:, idx]
        center = np.median(normal, axis=0)
        scale = np.maximum(np.quantile(normal, .75, axis=0) - np.quantile(normal, .25, axis=0), 1e-9)
        return Fitted(name, {"center": center, "scale": scale, "idx": idx},
                      lambda A: np.max((A[:, idx] - center) / scale, axis=1))
    if name == "isolation_forest":
        m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=1).fit(X[y == 0])
        return Fitted(name, m, lambda A: -m.score_samples(A))
    if name == "logistic":
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=0.3, class_weight=class_weight, max_iter=3000).fit(sc.transform(X), y)
        return Fitted(name, (sc, m), lambda A: m.decision_function(sc.transform(A)), lambda A: m.predict_proba(sc.transform(A))[:, 1])
    if name == "decision_tree":
        m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight=class_weight, random_state=SEED).fit(X, y)
        return Fitted(name, m, lambda A: _logit(m.predict_proba(A)[:, 1]), lambda A: m.predict_proba(A)[:, 1])
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=64, max_depth=8, min_samples_leaf=8, max_features="sqrt",
                                   class_weight=class_weight, random_state=SEED, n_jobs=1).fit(X, y)
        return Fitted(name, m, lambda A: _logit(m.predict_proba(A)[:, 1]), lambda A: m.predict_proba(A)[:, 1])
    if name == "extra_trees":
        m = ExtraTreesClassifier(n_estimators=64, max_depth=6, min_samples_leaf=8, max_features=1.0,
                                 class_weight=class_weight, random_state=SEED, n_jobs=1).fit(X, y)
        return Fitted(name, m, lambda A: _logit(m.predict_proba(A)[:, 1]), lambda A: m.predict_proba(A)[:, 1])
    if name == "hist_gb":
        hp = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
        hp.update(params)
        m = HistGradientBoostingClassifier(**hp, class_weight=class_weight, early_stopping=False, random_state=SEED).fit(X, y)
        return Fitted(name, m, lambda A: m.decision_function(A), lambda A: m.predict_proba(A)[:, 1])
    if name == "soft_vote":
        members = [fit_candidate(n, X, y, class_weight) for n in ("logistic", "extra_trees", "hist_gb")]
        prob = lambda A: np.mean([mm.proba(A) for mm in members], axis=0)
        return Fitted(name, members, lambda A: _logit(prob(A)), prob)
    raise ValueError(name)


def oof_scores(name, X, y, folds, judged, class_weight="balanced", params=None):
    """Out-of-fold raw scores. Each fold model is trained on judged rows of its training part."""
    out = np.full(len(y), np.nan)
    for train_mask, test_mask in folds:
        tr = train_mask & judged
        model = fit_candidate(name, X[tr], y[tr], class_weight, params)
        out[test_mask] = model.raw(X[test_mask])
    return out


# ----------------------------------------------------------------------------- calibration & thresholds
def fit_platt(raw, y):
    m = LogisticRegression(C=1e6, max_iter=2000).fit(np.asarray(raw).reshape(-1, 1), y)
    return {"method": "Platt logistic on out-of-fold development scores", "coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}


def apply_platt(raw, cal):
    z = cal["intercept"] + cal["coef"] * np.asarray(raw, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def _sweep(y, p):
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]
    return ps[last], tp[last], fp[last], int(ys.sum()), int(len(ys) - ys.sum())


def threshold_fpr_cap(y, p, cap):
    """Max F1 subject to FPR <= cap; ties -> fewer FP -> higher threshold. Candidates are observed scores."""
    th, tp, fp, P, N = _sweep(y, p)
    fn = P - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    ok = fp / max(N, 1) <= cap
    if not ok.any():
        return float(th[0]) + 1e-12
    best = max(((f1[i], -fp[i], th[i]) for i in np.flatnonzero(ok)))
    return float(best[2])


def threshold_recall(y, p, target):
    th, tp, fp, P, N = _sweep(y, p)
    ok = tp / max(P, 1) >= target
    return float(np.max(th[ok])) if ok.any() else float(np.min(th))


def mode_thresholds(y, p):
    """y, p: development OOF labels and calibrated scores of JUDGED rows."""
    standard = threshold_fpr_cap(y, p, FPR_CAP["standard"])
    unmanned = max(threshold_fpr_cap(y, p, FPR_CAP["unmanned"]), standard)
    screening = min(threshold_recall(y, p, SCREENING_RECALL), standard)
    return {"unmanned": unmanned, "standard": standard, "screening": screening}


MODE_RULES = {
    "unmanned": "Development OOF: max F1 subject to FPR <= 0.1% (precision-first, unattended hours)",
    "standard": "Development OOF: max F1 subject to FPR <= 1% (default)",
    "screening": "Development OOF: highest threshold keeping judged-row recall >= 95% (inspection candidate list)",
}


# ----------------------------------------------------------------------------- metrics
def decisions(score, judged, threshold):
    return judged & (np.asarray(score) >= threshold)


def row_metrics(y, score, judged, threshold):
    y = np.asarray(y).astype(int)
    pred = decisions(score, judged, threshold)
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    js = np.asarray(score)[judged]; jy = y[judged]
    out = {"rows": int(len(y)), "judged_rows": int(judged.sum()), "positive_rows": int(y.sum()), "threshold": float(threshold),
           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
           "false_positive_rate": fp / max(fp + tn, 1)}
    if 0 < jy.sum() < len(jy):
        out.update({"ap_judged": float(average_precision_score(jy, js)), "roc_auc_judged": float(roc_auc_score(jy, js))})
        if np.all((js >= 0) & (js <= 1)):
            out["brier_judged"] = float(brier_score_loss(jy, js))
    return out


def persistent_alarm(pred, segment_ids, judged, window=3, votes_required=2):
    """Replicates the deployed 2-of-3 rule: only judged rows vote; history restarts at every segment."""
    pred, judged = np.asarray(pred, bool), np.asarray(judged, bool)
    alarm = np.zeros(len(pred), bool)
    event = np.zeros(len(pred), bool)
    previous_segment, votes, previous_alarm = None, [], False
    for i, s in enumerate(segment_ids):
        if s != previous_segment:
            votes, previous_alarm, previous_segment = [], False, s
        if judged[i]:
            votes = (votes + [int(pred[i])])[-window:]
            alarm[i] = sum(votes) >= votes_required
        event[i] = alarm[i] and not previous_alarm
        previous_alarm = alarm[i]
    return alarm, event


def alarm_metrics(y, alarm, event, segment_ids, timestamps=None):
    y = np.asarray(y).astype(int)
    tp = int(np.sum(alarm & (y == 1))); fp = int(np.sum(alarm & (y == 0))); fn = int(np.sum(~alarm & (y == 1)))
    seg = pd.DataFrame({"s": segment_ids, "y": y, "a": alarm})
    per = seg.groupby("s", sort=False).agg(y=("y", "first"), a=("a", "any"))
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": int(np.sum(~alarm & (y == 0))),
           "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
           "normal_alarm_events": int(np.sum(event & (y == 0))), "normal_observed_seconds": float(np.sum(y == 0) * 0.1),
           "anomalous_segments_detected": int(per[per.y == 1].a.sum()), "anomalous_segments_total": int((per.y == 1).sum()),
           "normal_segments_with_alarm": int(per[per.y == 0].a.sum()), "normal_segments_total": int((per.y == 0).sum())}
    out["normal_alarm_events_per_observed_hour"] = out["normal_alarm_events"] / max(out["normal_observed_seconds"] / 3600, 1e-9)
    return out


def segment_bootstrap(y, score, judged, segment_ids, threshold, iterations=1000, seed=SEED):
    """Stratified (by label) resampling of whole segments. Conditional on the two source files."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); segment_ids = np.asarray(segment_ids)
    groups = {lab: [np.flatnonzero(segment_ids == s) for s in pd.unique(segment_ids[y == lab])] for lab in (0, 1)}
    keys = ["f1", "precision", "recall", "ap_judged"]
    res = {k: [] for k in keys}
    for _ in range(iterations):
        ix = np.concatenate([groups[l][j] for l in (0, 1) for j in rng.integers(len(groups[l]), size=len(groups[l]))])
        m = row_metrics(y[ix], np.asarray(score)[ix], np.asarray(judged)[ix], threshold)
        for k in keys:
            res[k].append(m.get(k, np.nan))
    return {"method": "label-stratified segment bootstrap, conditional on the two source recordings",
            "iterations": iterations, "anomalous_segments": len(groups[1]), "normal_segments": len(groups[0]),
            "bounds95": {k: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))] for k, v in res.items()}}


def select_candidate(results):
    """results: {name: {"f1":..,"ap":..,"fp":..}} on development OOF. Eligible only."""
    eligible = [n for n in results if CANDIDATE_BY_NAME[n]["eligible"]]
    return max(eligible, key=lambda n: (round(results[n]["f1"], 12), results[n]["ap"], -results[n]["fp"], -SIMPLICITY[n]))


SELECTION_RULE = ("Eligible candidate with the highest development OOF row F1 at its own standard threshold "
                  "(FPR <= 1%); ties -> higher OOF AP -> fewer FP -> simpler candidate")


# ----------------------------------------------------------------------------- export (stdlib JSON runtime)
def export_predictor(fitted):
    name, m = fitted.name, fitted.model
    if name == "rule_std":
        return {"kind": "robust", "center": m["center"].tolist(), "scale": m["scale"].tolist()}, [CENTERED[i] for i in m["idx"]]
    if name == "logistic":
        sc, lr = m
        return {"kind": "logistic", "center": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}, CENTERED
    if name in ("decision_tree", "random_forest", "extra_trees"):
        estimators = [m] if name == "decision_tree" else m.estimators_
        trees = []
        for est in estimators:
            t = est.tree_
            values = t.value[:, 0, :]
            trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
                          "threshold": t.threshold.tolist(), "probability": (values[:, 1] / values.sum(axis=1)).tolist()})
        return {"kind": "forest", "trees": trees}, CENTERED
    if name == "hist_gb":
        trees = []
        for predictors in m._predictors:
            nodes = predictors[0].nodes
            leaf = nodes["is_leaf"].astype(bool)
            left = nodes["left"].astype(np.int64); right = nodes["right"].astype(np.int64)
            feature = nodes["feature_idx"].astype(np.int64)
            trees.append({"left": np.where(leaf, -1, left).tolist(),
                          "right": np.where(leaf, -1, right).tolist(),
                          "feature": np.where(leaf, -2, feature).tolist(),
                          "threshold": np.where(leaf, -2.0, nodes["num_threshold"]).astype(float).tolist(),
                          "value": nodes["value"].astype(float).tolist(),
                          "cover": nodes["count"].astype(float).tolist()})
        return {"kind": "boosting", "baseline": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}, CENTERED
    if name == "soft_vote":
        members = []
        for member in m:
            predictor, _ = export_predictor(member)
            members.append(predictor)
        return {"kind": "ensemble", "combine": "mean_probability", "members": members}, CENTERED
    raise ValueError("Candidate is not deployable: " + name)
