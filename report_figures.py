"""Report figures that combine train.py / analysis.py outputs (no fitting).
Usage: python report_figures.py --output-dir ../evidence"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from protocol import korean_font

MODES = [("unmanned", "무인 감시"), ("standard", "표준 운전"), ("screening", "점검 선별")]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="../evidence")
    out = Path(parser.parse_args().output_dir).resolve(); korean_font()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8")); a = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    rows, pers, thr = m["test"]["rows"], m["test"]["persistent"], m["thresholds"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(3); w = .2
    series = [([rows[k]["precision"] for k, _ in MODES], "정밀도(표본)", "#1f7a8c"), ([rows[k]["recall"] for k, _ in MODES], "재현율(표본)", "#c0392b"),
              ([pers[k]["precision"] for k, _ in MODES], "정밀도(지속경보)", "#7fb7be"), ([pers[k]["recall"] for k, _ in MODES], "재현율(지속경보)", "#e59a8f")]
    for i, (vals, label, color) in enumerate(series):
        bars = axes[0].bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.2f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x, [f"{n}\n기준 {thr[k]:.3f}\n정상 경보진입 {pers[k]['normal_alarm_events']}건" for k, n in MODES])
    axes[0].set_ylim(0, 1.15); axes[0].legend(fontsize=8, ncol=2, loc="upper center")
    axes[0].set_title("시험 구간 운영 모드별 성능 (기준은 개발 OOF 규칙으로만 결정)")
    t = pd.read_csv(out / "screening_segment_ranking_test.csv")
    found = np.cumsum(t.label.to_numpy()); share = np.arange(1, len(t) + 1) / len(t)
    axes[1].plot(np.r_[0, share], np.r_[0, found / found[-1]], color="#7a4b07", lw=2, label="점검 선별 순위(시험)")
    axes[1].plot([0, .3], [0, .3], ls="--", color="grey", label="무작위 점검")
    k = a["inspection_ranking"]["test"]["inspect_top_n_to_cover_all"]
    axes[1].axvline(k / len(t), color="#333", ls=":", lw=1)
    axes[1].text(k / len(t) + .02, .35, f"상위 {k}개 구간({k / len(t):.1%}) 점검으로\n이상 구간 {int(found[-1])}개 모두 포함\n(개발 OOF: 상위 {a['inspection_ranking']['development_oof']['inspect_share_to_cover_all']:.1%}로 16/16)", fontsize=9)
    axes[1].set(xlabel="점검한 연속구간 비율(우선순위 순)", ylabel="포착한 이상 구간 비율", title="품질검사·점검 우선순위: 누적 포착 곡선", xlim=(0, .3), ylim=(0, 1.03))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "operating_modes.png", dpi=150); plt.close(fig)

    b = pd.read_csv(out / "calibration_bins.csv"); c = a["calibration_test_judged"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="완벽한 보정")
    for name, color, mk in [("보정 전(균형가중 확률)", "#9aa5ae", "s"), ("Platt 보정 후", "#087f8c", "o")]:
        s = b[b.calibration.eq(name)]
        ax.plot(s.mean_predicted, s.observed_anomaly_rate, marker=mk, color=color, label=name)
        for _, r in s.iterrows():
            if name.startswith("Platt"):
                ax.annotate(f"{int(r.rows)}", (r.mean_predicted, r.observed_anomaly_rate), textcoords="offset points", xytext=(5, -10), fontsize=7, color=color)
    ax.set(xlabel="예측 이상확률(구간 평균)", ylabel="실제 이상 비율", xlim=(0, 1), ylim=(0, 1.02),
           title=f"시험 신뢰도 곡선: Brier {c['brier_before']:.4f}→{c['brier_after']:.4f}, ECE {c['ece_before']:.3f}→{c['ece_after']:.3f}")
    ax.title.set_fontsize(9.5); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "calibration_reliability.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
