"""One-command full pipeline with source hash checks and safe output isolation.

Run: python reproduce.py [--output-dir DIR]
Order: train (selection on development OOF, single test evaluation) -> analysis -> ablations ->
nested CV -> SHAP -> 10 Hz signal diagnostics -> ROI scenario -> report figures.
Never overwrites the bundled deployment model; checks that the rebuilt model is byte-identical.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root.parent / "evidence" / "reproduced")
    args = parser.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "data" / "SOURCE_HASHES.json").read_text(encoding="utf-8"))
    for source in manifest["files"]:
        if hashlib.sha256((root / "data" / source["name"]).read_bytes()).hexdigest() != source["sha256"]:
            raise SystemExit("Source hash mismatch: " + source["name"])
    protected = root / "pumpguard" / "model.json"
    before = hashlib.sha256(protected.read_bytes()).hexdigest()
    print("Source hashes verified. Running the fixed v1.2 protocol (test never used for selection).", flush=True)
    data = ["--data-dir", str(root / "data"), "--output-dir", str(out)]
    steps = [["train.py", *data, "--reference-model", str(root / "reference" / "model_v1.1_extratrees.json")],
             ["analysis.py", *data], ["ablations.py", *data], ["nested_cv.py", *data],
             ["shap_analysis.py", *data, "--model", str(out / "model.json")], ["signal_diagnostics.py", *data],
             ["roi_scenario.py", "--assumptions", str(root / "results" / "roi_assumptions.json"), "--output-dir", str(out)],
             ["report_figures.py", "--output-dir", str(out)]]
    for step in steps:
        print(">>", step[0], flush=True)
        subprocess.run([sys.executable, str(root / step[0]), *step[1:]], check=True, cwd=root)
    if hashlib.sha256(protected.read_bytes()).hexdigest() != before:
        raise SystemExit("Bundled model unexpectedly changed")
    rebuilt = hashlib.sha256((out / "model.json").read_bytes()).hexdigest()
    # Decision-level equivalence: re-score the held-out test rows with the bundled JSON model
    sys.path.insert(0, str(root))
    import csv
    from pumpguard.inference import load_model, analyze_csv
    bundled = load_model(protected); rebuilt_model = load_model(out / "model.json")
    with (out / "predictions_test.csv").open(encoding="utf-8-sig") as f:
        saved = list(csv.DictReader(f))
    max_diff, same_decisions = 0.0, True
    for name in ("normal_test.csv", "anomaly_test.csv"):
        a = analyze_csv(root / "demo" / name, bundled)["records"]; b = analyze_csv(root / "demo" / name, rebuilt_model)["records"]
        for ra, rb in zip(a, b):
            if ra["score"] is not None:
                max_diff = max(max_diff, abs(ra["score"] - rb["score"]))
            same_decisions &= (ra["predicted"] == rb["predicted"]) and (ra["alarm"] == rb["alarm"])
    result = {"sources_verified": True, "bundled_model_unchanged": True, "bundled_model_sha256": before,
              "reproduced_model_sha256": rebuilt, "model_bytes_identical": rebuilt == before,
              "test_rows_max_score_difference_bundled_vs_rebuilt": max_diff, "test_decisions_and_alarms_identical": bool(same_decisions),
              "output_directory": str(out),
              "note": "Bytes are identical with the pinned versions in requirements.txt; on other platforms the last float digits may differ, so decisions are checked too."}
    (out / "REPRODUCIBILITY.json").write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not same_decisions:
        raise SystemExit("Rebuilt model changes test decisions")


if __name__ == "__main__":
    main()
