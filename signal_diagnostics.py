"""What does a 10 Hz row actually contain? Evidence for Chapter 1 and a
domain-feature check for Chapter 2/5.

Usage: python signal_diagnostics.py --data-dir data --output-dir ../evidence

Findings are computed, not asserted:
1. Sign/shape test: an RMS, peak or envelope value is never negative. If a channel
   is signed and centered, it is an instantaneous sample, not an envelope.
2. Shape of the current: a sinusoid sampled at random phase has an arcsine
   (U-shaped) distribution with excess kurtosis -1.5. Gaussian noise gives 0.
3. Spectrum inside 50-row (5 s) segments: with 10 Hz sampling only 0-5 Hz can be
   represented, so any line/inverter frequency above 5 Hz appears as an alias.
4. Data quality and source confounding (Chapter 1 tables): IQR outliers, anomalous rows inside
   the normal IQR fences, recorded decimal places, and per-segment means of the normal file.
Zero-crossing rate and sign imbalance of the current depend on the absolute offset of the
signal, so (like the level features) they are excluded from the model by the offset-invariance
rule in protocol.py; this script is descriptive only and fits no model.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pumpguard.features import FEATURE_NAMES, CHANNELS
from protocol import load_dataset, make_features, korean_font


def alias_features(d):
    rows = []
    for _, g in d.groupby("segment_id", sort=False):
        c = g.AI2_Current.to_numpy()
        for i in range(len(c)):
            w = c[max(0, i - 9):i + 1]
            zc = float(np.mean(np.signbit(w[1:]) != np.signbit(w[:-1]))) if len(w) > 1 else 0.0
            ac = 0.0
            if len(w) >= 3:
                a, b = w[:-1] - w.mean(), w[1:] - w.mean()
                den = np.sqrt((a * a).sum() * (b * b).sum())
                ac = float((a * b).sum() / den) if den > 1e-9 else 0.0
            rows.append([zc, ac, abs(float(np.mean(w > 0)) - .5)])
    return np.array(rows)


def data_quality(d, sources, data_dir):
    """Numbers quoted in Chapter 1 (tables 1, 4, 5)."""
    normal, anomalous = d[d.label == 0], d[d.label == 1]
    files = {}
    for source in sources:
        g = d[d.source_file.eq(source["name"])]
        size = g.groupby("segment_id").size()
        raw = pd.read_csv(next(Path(data_dir).rglob(source["name"])), dtype=str)
        files[source["name"]] = {
            "first_timestamp": g.TimeStamp.iloc[0], "last_timestamp": g.TimeStamp.iloc[-1],
            "raw_rows": source["raw_rows"], "deduplicated_rows": source["deduplicated_rows"],
            "segments": int(len(size)), "segment_rows_min": int(size.min()), "segment_rows_max": int(size.max()),
            "gaps_over_0.15s": int(len(size) - 1),
            "decimal_places_vibration": {c: {int(k): int(v) for k, v in raw[c].str.split(".").str[1].str.len().value_counts().items()}
                                         for c in CHANNELS[:2]}}
    fences, inside = {}, np.ones(len(anomalous), bool)
    for c in CHANNELS:
        q1, q3 = normal[c].quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        fences[c] = {"normal_outlier_share": float(((normal[c] < lo) | (normal[c] > hi)).mean()), "fence": [float(lo), float(hi)]}
        inside &= anomalous[c].between(lo, hi).to_numpy()
    full = normal.groupby("segment_id").filter(lambda g: len(g) == 50).groupby("segment_id")[list(CHANNELS[:2])].mean().abs()
    return {"files": files, "iqr_1.5": fences,
            "anomalous_rows_inside_all_normal_fences": float(inside.mean()),
            "normal_50row_segment_vibration_mean_abs": {"segments": int(len(full)),
                                                        "median": full.median().to_dict(), "max": full.max().to_dict()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="../evidence")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    korean_font()
    d, sources = load_dataset(args.data_dir)
    X, history = make_features(d)
    y = d.label.to_numpy()
    shape = {}
    for label in (0, 1):
        s = d[d.label == label]
        shape[label] = {c: {"negative_share": float((s[c] < 0).mean()), "mean": float(s[c].mean()),
                            "excess_kurtosis": float(s[c].kurt()), "min": float(s[c].min()), "max": float(s[c].max())}
                        for c in CHANNELS}
    freqs = np.fft.rfftfreq(50, 0.1)
    spectra = {}
    for label in (0, 1):
        spectra[label] = {}
        for c in CHANNELS:
            acc = []
            for _, g in d[d.label == label].groupby("segment_id"):
                if len(g) == 50:
                    x = g[c].to_numpy() - g[c].mean()
                    power = np.abs(np.fft.rfft(x * np.hanning(50))) ** 2
                    if power.sum() > 0:
                        acc.append(power / power.sum())
            spectra[label][c] = {"segments": len(acc), "mean_power_share": np.mean(acc, axis=0).tolist()}
    peaks = {label: {c: sorted([(float(freqs[i]), float(v)) for i, v in enumerate(spectra[label][c]["mean_power_share"])],
                              key=lambda t: -t[1])[:3] for c in CHANNELS} for label in (0, 1)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, color, text in [(0, "#1f7a8c", "정상"), (1, "#c0392b", "이상")]:
        s = d[d.label == label]
        axes[0].hist(s.AI2_Current, bins=40, density=True, alpha=.55, color=color, label=f"{text} (첨도 {shape[label]['AI2_Current']['excess_kurtosis']:+.2f})")
        axes[1].plot(freqs, spectra[label]["AI2_Current"]["mean_power_share"], marker="o", color=color, label=text)
        axes[2].hist(s.AI0_Vibration, bins=60, density=True, alpha=.55, color=color, label=f"{text} (음수 {shape[label]['AI0_Vibration']['negative_share']:.0%})")
    axes[0].set(title="전류 값 분포: 정상은 ± 양봉형(교류 순시값의 특징)", xlabel="AI2_Current (원 단위)", ylabel="밀도")
    axes[0].legend(fontsize=8, title="첨도: 정현파 -1.5 / 가우시안 0", title_fontsize=8)
    axes[1].set(title="50행(5초) 구간 전류 평균 스펙트럼", xlabel="주파수 (Hz, 10Hz 표본 → 0~5Hz만 표현)", ylabel="전력 비율"); axes[1].legend()
    axes[2].set(title="AI0 진동값은 부호가 있음 → RMS·포락선 아님", xlabel="AI0_Vibration (원 단위)"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "signal_diagnostics.png", dpi=150); plt.close(fig)

    y = d.label.to_numpy()
    extra = alias_features(d)
    names = ["current_zero_cross_rate", "current_lag1_autocorr", "current_sign_imbalance"]
    full = history >= 10
    alias_means = {int(label): dict(zip(names, np.mean(extra[(y == label) & full], axis=0).round(4).tolist())) for label in (0, 1)}
    summary = {"shape": {int(k): v for k, v in shape.items()}, "spectrum_frequency_hz": freqs.tolist(),
               "spectrum_top3_peaks": {int(k): v for k, v in peaks.items()},
               "spectrum_mean_power_share": {int(k): {c: v["mean_power_share"] for c, v in s.items()} for k, s in spectra.items()},
               "alias_feature_means_full_window": alias_means,
               "data_quality": data_quality(d, sources, args.data_dir),
               "decision": "Descriptive only. Offset-dependent signal features are excluded from the model by design (label = file = date).",
               "limits": "Channel units, DAQ sampling settings and motor drive type are not in the dataset; the alias explanation is a hypothesis to confirm on site."}
    (out / "signal_diagnostics.json").write_bytes(json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps({"peaks_current": {k: v["AI2_Current"] for k, v in summary["spectrum_top3_peaks"].items()}, "alias_means": alias_means}, ensure_ascii=False))


if __name__ == "__main__":
    main()
