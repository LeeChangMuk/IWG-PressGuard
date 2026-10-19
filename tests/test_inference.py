"""Meaningful safety, causality and portable-prediction contract tests."""
import copy
import csv
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pumpguard.features import CHANNELS,FEATURE_NAMES,CausalFeatureExtractor
from pumpguard.inference import load_model,validate_model,predict_features,raw_score,StreamAnalyzer,analyze_csv,mode_threshold,OPERATING_MODES
from pumpguard.inference import EVIDENCE_FEATURES,feature_contributions,expected_raw_score
import hashlib

# SHA-256 of the canonical decision core (trees, calibration, thresholds, alarm policy) of the
# frozen v1.2 model. Any retraining or threshold edit must be deliberate and change this value.
V12_DECISION_CORE_SHA256="6684b74799838c927b94fa0da0ffe18ef5221f738547f3ff886b54fd543eed85"


def decision_core_sha256(model):
    core={k:model[k] for k in ("feature_indices","predictor","calibration","threshold","alarm_policy","operating_modes")}
    return hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_model(ROOT/"pumpguard"/"model.json")
        with (ROOT/"demo"/"normal_test.csv").open(encoding="utf-8-sig") as f:
            cls.rows=list(csv.DictReader(f))[:40]

    def stamp(self,i):return datetime(2022,1,1)+timedelta(seconds=.1*i)
    def values(self,i):return [.03*math.sin(i),.05*math.cos(i),100*math.sin(i*.3)]

    def test_prefix_causality(self):
        a=CausalFeatureExtractor();b=CausalFeatureExtractor();expected=[]
        for i in range(30):expected.append(a.update(self.values(i),self.stamp(i))["features"])
        actual=[b.update(self.values(i),self.stamp(i))["features"] for i in range(15)]
        for i in range(15,30):b.update([500,300,2000],self.stamp(i))
        self.assertEqual(expected[:15],actual)
        self.assertEqual(len(expected[0]),27)

    def test_model_features_are_offset_invariant(self):
        # label = file = date, so a per-file sensor offset must not change any model input
        idx=self.model["feature_indices"];a=CausalFeatureExtractor();b=CausalFeatureExtractor()
        for i in range(12):
            fa=a.update(self.values(i),self.stamp(i))["features"]
            v=self.values(i);fb=b.update([v[0]+.7,v[1]-.4,v[2]+55.],self.stamp(i))["features"]
            for k in idx:self.assertAlmostEqual(fa[k],fb[k],places=9)

    def test_gap_resets_exactly(self):
        a=CausalFeatureExtractor()
        for i in range(9):a.update(self.values(i),self.stamp(i))
        after=a.update([.1,.2,50],self.stamp(30))
        fresh=CausalFeatureExtractor().update([.1,.2,50],self.stamp(30))
        self.assertEqual(after["features"],fresh["features"])
        self.assertEqual(after["n_history"],1);self.assertTrue(after["gap_reset"])

    def test_invalid_input_never_normal_and_resets(self):
        for bad in [float("nan"),float("inf"),None,"bad",1e308]:
            stream=StreamAnalyzer(self.model)
            for i in range(3):stream.update(self.values(i),self.stamp(i))
            result=stream.update([bad,.1,100],self.stamp(3))
            self.assertEqual(result["status"],"INVALID");self.assertIsNone(result["score"])
            self.assertEqual(stream.update(self.values(4),self.stamp(4))["status"],"WARMUP")

    def test_bad_time_and_oversampling_are_invalid(self):
        for stamp in ["", "not a timestamp",self.stamp(-1),self.stamp(.5)]:
            stream=StreamAnalyzer(self.model);stream.update(self.values(0),self.stamp(0))
            self.assertEqual(stream.update(self.values(1),stamp)["status"],"INVALID")

    def test_duplicate_is_excluded_without_history_growth(self):
        stream=StreamAnalyzer(self.model);stream.update(self.values(0),self.stamp(0))
        duplicate=stream.update(self.values(0),self.stamp(0))
        self.assertTrue(duplicate["duplicate"]);self.assertIsNone(duplicate["score"])
        self.assertEqual(stream.update(self.values(1),self.stamp(1))["n_history"],2)

    def test_frozen_stays_review_then_requires_warmup(self):
        stream=StreamAnalyzer(self.model)
        for i in range(10):result=stream.update([0,.05*math.cos(i),50+i],self.stamp(i))
        self.assertEqual(result["status"],"REVIEW");self.assertEqual(len(stream.votes),0)
        self.assertEqual(stream.update([0,.2,100],self.stamp(10))["status"],"REVIEW")
        unblocked=stream.update([.01,.2,100],self.stamp(11))
        self.assertEqual(unblocked["status"],"WARMUP");self.assertEqual(unblocked["n_history"],1)

    def test_ood_cannot_vote_for_future_alarm(self):
        stream=StreamAnalyzer(self.model)
        limit=self.model["review_absolute_limits"][0]
        result=stream.update([limit*2,.1,100],self.stamp(0))
        self.assertEqual(result["status"],"REVIEW");self.assertEqual(len(stream.votes),0)
        self.assertFalse(result["alarm"])
        nextrow=stream.update(self.values(1),self.stamp(1))
        self.assertEqual(nextrow["status"],"WARMUP");self.assertEqual(nextrow["n_history"],1)

    def test_labels_and_unrelated_fields_do_not_change_output(self):
        with tempfile.TemporaryDirectory() as folder:
            scores=[]
            for label in [0,1]:
                path=Path(folder)/f"label{label}.csv"
                with path.open("w",newline="",encoding="utf-8") as f:
                    fields=["TimeStamp",*CHANNELS,"Equipment_state","filename","row_index"]
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                    for i,row in enumerate(self.rows):writer.writerow({k:row[k] for k in ["TimeStamp",*CHANNELS]}|{"Equipment_state":label,"filename":str(label),"row_index":i+1000*label})
                scores.append([r["score"] for r in analyze_csv(path,self.model)["records"]])
            self.assertEqual(scores[0],scores[1])

    def test_malformed_csv_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"bad.csv"
            for text in ["TimeStamp,AI0_Vibration\n2022-01-01,1\n",",".join(["TimeStamp",*CHANNELS])+"\n",
                         "TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current,AI2_Current\n2022-01-01,1,2,3,4\n"]:
                p.write_text(text,encoding="utf-8")
                with self.assertRaises(ValueError):analyze_csv(p,self.model)

    def test_schema_rejects_nonfinite_threshold_and_cycles(self):
        for key,value in [("threshold",float("nan")),("expected_sampling_hz",20),
                          ("feature_indices",[0,0]),("review_absolute_limits",[-1,2,3])]:
            model=copy.deepcopy(self.model);model[key]=value
            with self.assertRaises(ValueError):validate_model(model)
        model=copy.deepcopy(self.model);model["predictor"]["trees"][0]["left"][0]=0
        with self.assertRaises(ValueError):validate_model(model)

    def test_tree_boundary_uses_float32_like_sklearn(self):
        threshold=struct.unpack("f",struct.pack("f",.1))[0]
        value=math.nextafter(threshold,math.inf)
        tree={"left":[1,-1,-1],"right":[2,-1,-1],"feature":[0,-2,-2],
              "threshold":[threshold,-2.,-2.],"probability":[.5,.2,.8]}
        model={"feature_indices":[0],"predictor":{"kind":"forest","trees":[tree]}}
        vector=[0.]*len(FEATURE_NAMES);vector[0]=value
        self.assertAlmostEqual(raw_score(vector,model),math.log(.2/.8))
        vector[0]=threshold+1e-6
        self.assertAlmostEqual(raw_score(vector,model),math.log(.8/.2))

    def test_persistence_is_distinct_from_row_prediction(self):
        model=copy.deepcopy(self.model);model["threshold"]=0.
        stream=StreamAnalyzer(model)
        results=[stream.update(self.values(i),self.stamp(i)) for i in range(4)]
        # v1.2: rows with fewer than 3 samples are not judged, do not predict and do not vote
        self.assertEqual([r["status"] for r in results[:2]],["WARMUP","WARMUP"])
        self.assertFalse(results[0]["predicted"]);self.assertFalse(results[1]["predicted"])
        self.assertTrue(results[2]["predicted"]);self.assertFalse(results[2]["alarm"])
        self.assertTrue(results[3]["predicted"]);self.assertTrue(results[3]["alarm"])

    def test_warmup_rows_never_vote(self):
        model=copy.deepcopy(self.model);model["threshold"]=0.
        stream=StreamAnalyzer(model)
        for i in range(2):stream.update(self.values(i),self.stamp(i))
        self.assertEqual(len(stream.votes),0)

    def test_boosting_runtime_matches_tree_walk(self):
        tree={"left":[1,-1,-1],"right":[2,-1,-1],"feature":[0,-2,-2],"threshold":[.5,-2.,-2.],"value":[0.,-1.25,2.5],"cover":[10.,6.,4.]}
        model={"feature_indices":[0],"predictor":{"kind":"boosting","baseline":.1,"trees":[tree,tree]}}
        vector=[0.]*len(FEATURE_NAMES);vector[0]=.5
        self.assertAlmostEqual(raw_score(vector,model),.1-2.5)
        vector[0]=.5000001
        self.assertAlmostEqual(raw_score(vector,model),.1+5.)

    def test_v12_decision_core_is_frozen(self):
        self.assertEqual(decision_core_sha256(self.model),V12_DECISION_CORE_SHA256)
        self.assertEqual(self.model["model_version"],"1.2.0")
        self.assertEqual(self.model["predictor"]["kind"],"boosting")

    def test_operating_modes_are_ordered_and_standard_is_frozen(self):
        t={m:mode_threshold(self.model,m) for m in OPERATING_MODES}
        self.assertEqual(t["standard"],self.model["threshold"])
        self.assertLess(t["screening"],t["standard"]);self.assertLess(t["standard"],t["unmanned"])
        with self.assertRaises(ValueError):mode_threshold(self.model,"turbo")
        for bad in [{"screening":.9},{"unmanned":.01},{"standard":.5}]:
            model=copy.deepcopy(self.model)
            for k,v in bad.items():model["operating_modes"][k]["threshold"]=v
            with self.assertRaises(ValueError):validate_model(model)

    def test_mode_changes_decision_only_not_scores(self):
        path=ROOT/"demo"/"anomaly_test.csv";runs={m:analyze_csv(path,self.model,m) for m in OPERATING_MODES}
        scores={m:[r["score"] for r in runs[m]["records"]] for m in runs}
        self.assertEqual(scores["unmanned"],scores["standard"]);self.assertEqual(scores["standard"],scores["screening"])
        alarms={m:runs[m]["summary"]["alarm_rows"] for m in runs}
        self.assertLessEqual(alarms["unmanned"],alarms["standard"]);self.assertLessEqual(alarms["standard"],alarms["screening"])
        for m in runs:
            self.assertEqual(runs[m]["metadata"]["operating_mode"],m)
            self.assertEqual(runs[m]["metadata"]["threshold"],mode_threshold(self.model,m))
        ratios=[runs[m]["summary"]["flag_ratio"] for m in ("unmanned","standard","screening")]
        self.assertEqual(ratios,sorted(ratios))

    def always_flag(self):
        model=copy.deepcopy(self.model);model["threshold"]=0.
        return model

    def test_alarm_evidence_is_offset_invariant(self):
        # evidence text must not reintroduce the per-file offset that the model inputs exclude
        inputs={FEATURE_NAMES[i] for i in self.model["feature_indices"]}
        self.assertTrue(all(name in inputs for name,_ in EVIDENCE_FEATURES))
        model=self.always_flag();a=StreamAnalyzer(model);b=StreamAnalyzer(model)
        for i in range(30):
            v=self.values(i);ra=a.update(v,self.stamp(i));rb=b.update([v[0]+.7,v[1]-.4,v[2]+55.],self.stamp(i))
            self.assertEqual((ra["status"],ra["reason"]),(rb["status"],rb["reason"]))

    def test_lower_vibration_still_guidance_is_text_only(self):
        for amplitude,expected in [(.01,True),(.2,False)]:
            stream=StreamAnalyzer(self.always_flag())
            records=[stream.update([.03*math.sin(i),amplitude*math.cos(i),100*math.sin(i*.3)],self.stamp(i)) for i in range(12)]
            flagged=[r for r in records if r["status"] in ("WATCH","ALARM")]
            self.assertEqual(len(flagged),10)
            self.assertTrue(all(r["lower_sensor_check"]==expected for r in flagged))
            self.assertEqual(all("하부 센서" in r["reason"] for r in flagged),expected)
            self.assertFalse(any(r["lower_sensor_check"] for r in records if r["status"]=="WARMUP"))

    def test_shap_contributions_are_additive(self):
        extractor=CausalFeatureExtractor();expected=expected_raw_score(self.model)
        for i,row in enumerate(self.rows[:12]):
            features=extractor.update([row[c] for c in CHANNELS],row["TimeStamp"])["features"]
            total=sum(v for _,v in feature_contributions(features,self.model))
            self.assertAlmostEqual(total+expected,raw_score(features,self.model),places=9)

    def test_shap_evidence_only_at_alarm_entry(self):
        records=analyze_csv(ROOT/"demo"/"anomaly_test.csv",self.model)["records"]
        events=[r for r in records if r["alarm_event"]]
        self.assertEqual(len(events),10)
        for r in records:
            if r["alarm_event"]:
                self.assertTrue(r["contributions"] and all(v>0 for _,v in r["contributions"]))
                self.assertIn("주요 기여(SHAP",r["reason"])
            else:
                self.assertEqual(r["contributions"],[])

    def test_segment_priority_matches_reported_test_ranking(self):
        # analysis.py: on the test split the 5 anomalous segments rank 1,2,3,4,6 of 124 (screening)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"test_split.csv"
            with path.open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f);writer.writerow(["TimeStamp",*CHANNELS])
                for name in ("normal_test.csv","anomaly_test.csv"):
                    with (ROOT/"demo"/name).open(encoding="utf-8-sig") as src:
                        for row in csv.DictReader(src):writer.writerow([row["TimeStamp"],*(row[c] for c in CHANNELS)])
            segments=analyze_csv(path,self.model,"screening")["segments"]
        self.assertEqual(len(segments),124)
        self.assertEqual([s["rank"] for s in segments if s["start"].startswith("2022-07-17")],[1,2,3,4,6])


if __name__=="__main__":unittest.main()
