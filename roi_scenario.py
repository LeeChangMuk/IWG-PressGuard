"""Assumption-based downtime-defence scenario for Chapter 4.

Usage: python roi_scenario.py --assumptions results/roi_assumptions.json --output-dir ../evidence

Every input is an ASSUMPTION until replaced by values confirmed at the site
(현장 A). Detection proxies come from the internal KAMP test split and are NOT
field failure-detection rates. Edit the JSON after the site visit and re-run.

  annual_defence  = stops_per_year * loss_per_stop * detect_share * avoid_share
  checks_per_week = site_standard_checks_per_week * mode_alarm_ratio_vs_standard[mode]
  alarm_check     = checks_per_week * 52 * hours_per_check * wage_per_hour
  net             = annual_defence - alarm_check
  break_even      = annual_defence / (52 * hours_per_check * wage_per_hour)   [checks per week at net = 0]
  payback_months  = adoption_cost / (net / 12)

The mode ratio is measured (normal alarm entries on the test split: unmanned 3, standard 10,
screening 24), so a more sensitive mode always carries a proportionally larger checking load.
The site's absolute alarm frequency is not: the KAMP normal record is a set of <= 5 s fragments
and must be re-measured after the site re-calibration step.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json


def evaluate(s, cost, ratio):
    defence = s["stops_per_year"] * s["loss_per_stop_krw"] * s["detect_share"] * s["avoid_share"]
    per_week = 52 * s["hours_per_check"] * s["wage_per_hour_krw"]
    checks = s["site_standard_checks_per_week"] * ratio[s["mode"]]
    net = defence - checks * per_week
    return {"alarm_checks_per_week": checks, "annual_defence_krw": defence, "annual_alarm_check_krw": checks * per_week,
            "annual_net_krw": net, "break_even_checks_per_week": defence / per_week,
            "payback_months": (cost / (net / 12)) if net > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assumptions", default=str(Path(__file__).resolve().parent / "results" / "roi_assumptions.json"))
    parser.add_argument("--output-dir", default="../evidence")
    args = parser.parse_args()
    a = json.loads(Path(args.assumptions).read_text(encoding="utf-8"))
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    ratio = a["mode_alarm_ratio_vs_standard"]
    rows = []
    for (name, s), (cname, cost) in itertools.product(a["scenarios"].items(), a["adoption_cost_krw"].items()):
        rows.append({"scenario": name, "adoption_case": cname, "adoption_cost_krw": cost, **s, **evaluate(s, cost, ratio)})
    ceiling = dict(a["scenarios"]["base"], detect_share=1.0, avoid_share=1.0, site_standard_checks_per_week=0)
    grid = []
    for loss in a["sensitivity"]["loss_per_stop_krw"]:
        for detect in a["sensitivity"]["detect_share"]:
            s = dict(a["scenarios"]["base"], loss_per_stop_krw=loss, detect_share=detect)
            grid.append({"loss_per_stop_krw": loss, "detect_share": detect, **evaluate(s, a["adoption_cost_krw"]["existing_sensors"], ratio)})
    with (out / "roi_scenarios.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {"status": a["status"], "formula": __doc__.split("\n\n")[3].strip(), "mode_alarm_ratio_vs_standard": ratio,
              "scenarios": rows,
              "upper_bound_perfect_detection_and_avoidance": evaluate(ceiling, a["adoption_cost_krw"]["existing_sensors"], ratio),
              "sensitivity_existing_sensors": grid, "sources_of_assumptions": a["notes"]}
    (out / "roi_scenarios.json").write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    for r in rows:
        pm = "n/a" if r["payback_months"] is None else f"{r['payback_months']:.2f}"
        print(f"{r['scenario']:12s} {r['adoption_case']:18s} checks/wk {r['alarm_checks_per_week']:.1f} (break-even {r['break_even_checks_per_week']:.1f}) "
              f"defence {r['annual_defence_krw']/1e4:,.0f}만 check {r['annual_alarm_check_krw']/1e4:,.1f}만 net {r['annual_net_krw']/1e4:,.0f}만 payback {pm}개월")
    print("upper bound", {k: round(v/1e4) if isinstance(v, float) and k.endswith('krw') else v for k, v in result["upper_bound_perfect_detection_and_avoidance"].items()})


if __name__ == "__main__":
    main()
