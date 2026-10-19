"""Causal, gap-aware signal features. Runtime uses only Python standard library.

Time is used only to reset history; neither timestamps nor segment length enter
the predictor. Physical units and AI0/AI1 sensor positions require site mapping.
"""
from collections import deque
from datetime import datetime
import math
import statistics

CHANNELS = ("AI0_Vibration", "AI1_Vibration", "AI2_Current")
FEATURE_NAMES = list(CHANNELS)
for channel in CHANNELS:
    FEATURE_NAMES.extend(channel + suffix for suffix in
                         ("_abs", "_rms", "_std", "_residual_abs", "_mean_abs", "_peak_abs", "_diff_abs"))
FEATURE_NAMES += ["vibration_correlation", "vibration_std_log_ratio", "vibration_rms_log_ratio"]
FEATURE_VERSION = "causal27-v1"


def parse_timestamp(value):
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("TimeStamp is empty")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class CausalFeatureExtractor:
    def __init__(self, max_window=10, max_gap_seconds=0.15):
        self.max_window = int(max_window)
        self.max_gap_seconds = float(max_gap_seconds)
        if self.max_window < 2:
            raise ValueError("max_window must be >= 2")
        self.history = deque(maxlen=self.max_window)
        self.previous_timestamp = None

    def reset(self):
        self.history.clear()
        self.previous_timestamp = None

    def update(self, values, timestamp):
        values = tuple(float(v) for v in values)
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("Three finite sensor values are required")
        timestamp = parse_timestamp(timestamp)
        gap_reset = False
        dt = None
        if self.previous_timestamp is not None:
            dt = (timestamp - self.previous_timestamp).total_seconds()
            gap_reset = dt <= 0 or dt > self.max_gap_seconds
            if gap_reset:
                self.history.clear()
        self.previous_timestamp = timestamp
        self.history.append(values)
        columns = list(zip(*self.history))
        means = [statistics.fmean(x) for x in columns]
        rms = [math.sqrt(statistics.fmean(v*v for v in x)) for x in columns]
        stds = [math.sqrt(statistics.fmean((v-m)**2 for v in x)) for x, m in zip(columns, means)]
        result = list(values)
        for i, column in enumerate(columns):
            result.extend((abs(values[i]), rms[i], stds[i], abs(values[i]-means[i]),
                           abs(means[i]), max(abs(v) for v in column),
                           abs(column[-1]-column[-2]) if len(column) > 1 else 0.0))
        covariance = statistics.fmean((a-means[0])*(b-means[1]) for a,b in zip(columns[0],columns[1]))
        corr = covariance/(stds[0]*stds[1]) if stds[0]*stds[1] > 1e-12 else 0.0
        result.extend((max(-1.0,min(1.0,corr)), math.log((stds[0]+1e-6)/(stds[1]+1e-6)),
                       math.log((rms[0]+1e-6)/(rms[1]+1e-6))))
        if not all(math.isfinite(v) for v in result):
            raise ValueError("Sensor values overflow the supported numeric range")
        frozen = [CHANNELS[i] for i,c in enumerate(columns)
                  if len(c) >= self.max_window and max(c)-min(c) <= 1e-12]
        return {"features":result,"n_history":len(self.history),"gap_reset":gap_reset,
                "dt_seconds":dt,"frozen_channels":frozen}
