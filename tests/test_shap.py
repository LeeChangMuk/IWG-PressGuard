"""The shipped JSON trees must equal the deterministic sklearn refit, and SHAP computed from the
JSON trees must equal SHAP computed on sklearn. Skipped when analysis packages are absent."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import numpy as np
    import shap
    import shap_analysis as sa
    from protocol import load_dataset, make_features, fit_candidate, CENTERED, MIN_HISTORY
    from pumpguard.inference import load_model, raw_score, feature_contributions, expected_raw_score
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "shap/scikit-learn not installed (runtime EXE does not need them)")
class ShapTests(unittest.TestCase):
    def test_json_model_and_shap_match_sklearn(self):
        d, _ = load_dataset(ROOT / "data")
        X, h = make_features(d)
        y = d.label.to_numpy(); dev = d.split.eq("development").to_numpy(); j = h >= MIN_HISTORY
        model = load_model(ROOT / "pumpguard" / "model.json")
        sk = fit_candidate(model["selected_candidate"], X[dev & j][:, CENTERED], y[dev & j])
        sample = X[~dev & j][:300]
        ours_raw = np.array([raw_score(r.tolist(), model) for r in sample])
        self.assertLess(np.max(np.abs(ours_raw - sk.raw(sample[:, CENTERED]))), 1e-9)
        ref = shap.TreeExplainer(sk.model).shap_values(sample[:, CENTERED])
        ours = shap.TreeExplainer(sa.json_tree_model(model)).shap_values(sample[:, CENTERED], check_additivity=False)
        self.assertLess(np.max(np.abs(ref - ours)), 1e-9)

    def test_runtime_alarm_evidence_equals_shap(self):
        # the EXE's standard-library TreeSHAP (alarm evidence) must equal shap.TreeExplainer
        d, _ = load_dataset(ROOT / "data")
        X, h = make_features(d)
        sample = X[d.split.eq("test").to_numpy() & (h >= MIN_HISTORY)][::40]
        model = load_model(ROOT / "pumpguard" / "model.json")
        ref = shap.TreeExplainer(sa.json_tree_model(model)).shap_values(sample[:, CENTERED], check_additivity=False)
        ours = np.array([[v for _, v in feature_contributions(r.tolist(), model)] for r in sample])
        self.assertLess(np.max(np.abs(ref - ours)), 1e-9)
        self.assertLess(abs(expected_raw_score(model) - float(np.ravel(shap.TreeExplainer(sa.json_tree_model(model)).expected_value)[0])), 1e-9)


if __name__ == "__main__":
    unittest.main()
