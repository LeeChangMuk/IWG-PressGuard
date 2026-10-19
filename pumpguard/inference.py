"""Portable inference and read-only CSV replay. Never controls equipment."""
import csv
import json
import math
import struct
from collections import deque
from pathlib import Path
import time
from .features import CHANNELS, FEATURE_NAMES, FEATURE_VERSION, CausalFeatureExtractor, parse_timestamp

STATUSES = ("WARMUP", "NORMAL", "WATCH", "ALARM", "REVIEW", "INVALID")
# Operating modes share one predictor; only the decision threshold differs. All three thresholds
# are chosen by train.py from development out-of-fold predictions (rules recorded in
# model["operating_modes"]); the test split never selects them.
OPERATING_MODES = ("unmanned", "standard", "screening")
MODE_LABELS = {"unmanned": "무인 24/7 감시", "standard": "표준 운전", "screening": "정기점검 선별"}
FEATURE_LABELS = {
    "AI0_Vibration_std": "AI0 진동 표준편차", "AI0_Vibration_residual_abs": "AI0 진동 순간편차", "AI0_Vibration_diff_abs": "AI0 진동 변화량",
    "AI1_Vibration_std": "AI1 진동 표준편차", "AI1_Vibration_residual_abs": "AI1 진동 순간편차", "AI1_Vibration_diff_abs": "AI1 진동 변화량",
    "AI2_Current_std": "전류 표준편차", "AI2_Current_residual_abs": "전류 순간편차", "AI2_Current_diff_abs": "전류 변화량",
    "vibration_correlation": "AI0-AI1 진동 상관", "vibration_std_log_ratio": "AI0/AI1 진동 SD 로그비",
}
# Evidence text only (never changes a score, decision or alarm). Offset-invariant model inputs are
# compared with the normal-training p99; level features such as RMS are not shown as evidence.
EVIDENCE_FEATURES = (("AI0_Vibration_std", "AI0 진동 변동"), ("AI1_Vibration_std", "AI1 진동 변동"),
                     ("AI2_Current_std", "전류 변동"), ("AI2_Current_diff_abs", "전류 급변"))
# Out-of-sample error analysis (analysis.py depth-2 slice, report ch.3): 70% of false alarms occurred
# while the lower vibration channel was almost still (AI1 window SD <= 0.02641, the slice's split value;
# 2.4% of normal rows).
LOWER_VIBRATION_STILL_STD = 0.02641


def sigmoid(x):
    if x >= 0:
        return 1.0/(1.0+math.exp(-min(x,700.0)))
    z=math.exp(max(x,-700.0))
    return z/(1.0+z)


def load_model(path):
    model=json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("schema_version") != 2 or model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported model/feature version")
    if model.get("feature_names") != FEATURE_NAMES or model.get("channels") != list(CHANNELS):
        raise ValueError("Model input schema does not match this application")
    try:
        validate_model(model)
    except (KeyError,TypeError,IndexError,OverflowError) as exc:
        raise ValueError("Incomplete or malformed model") from exc
    return model


def validate_model(model):
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    indices=model["feature_indices"]
    if not indices or len(indices)!=len(set(indices)) or not all(type(i) is int and 0<=i<len(FEATURE_NAMES) for i in indices):
        raise ValueError("Invalid feature indices")
    if not finite(model["threshold"]) or not 0<=model["threshold"]<=1:
        raise ValueError("Invalid decision threshold")
    if model["max_window"]!=10 or model["expected_sampling_hz"]!=10 or model["expected_sample_interval_seconds"]!=.1 or model["max_gap_seconds"]!=.15:
        raise ValueError("Unsupported sampling/window contract")
    policy=model["alarm_policy"]
    if policy!={"window":3,"votes_required":2,"minimum_history":3}:
        raise ValueError("Unsupported alarm policy")
    modes=model.get("operating_modes")
    if modes is not None:
        if not isinstance(modes,dict) or set(modes)!=set(OPERATING_MODES):
            raise ValueError("operating_modes must define unmanned, standard and screening")
        for name,spec in modes.items():
            if not finite(spec["threshold"]) or not 0<=spec["threshold"]<=1:
                raise ValueError("Invalid operating-mode threshold: "+name)
        if modes["standard"]["threshold"]!=model["threshold"]:
            raise ValueError("Standard mode must use the frozen model threshold")
        if not modes["screening"]["threshold"]<=modes["standard"]["threshold"]<=modes["unmanned"]["threshold"]:
            raise ValueError("Mode thresholds must be ordered screening <= standard <= unmanned")
    limits=model["review_absolute_limits"]
    if len(limits)!=3 or not all(finite(v) and v>0 for v in limits):
        raise ValueError("Invalid review limits")
    if not all(finite(model["calibration"][k]) for k in ("coef","intercept")):
        raise ValueError("Invalid calibration parameters")
    validate_predictor(model["predictor"],len(indices))


def validate_predictor(predictor,dimension,depth=0):
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    kind=predictor["kind"]
    if kind in ("robust","logistic"):
        for key in ("center","scale"):
            if len(predictor[key])!=dimension or not all(finite(v) for v in predictor[key]):
                raise ValueError("Predictor vector length or value invalid")
        if not all(v>0 for v in predictor["scale"]):raise ValueError("Nonpositive scale")
        if kind=="logistic" and (len(predictor["coef"])!=dimension or not all(finite(v) for v in predictor["coef"]) or not finite(predictor["intercept"])):
            raise ValueError("Invalid logistic coefficients")
    elif kind in ("forest","boosting"):
        if not predictor["trees"]:raise ValueError("Empty tree ensemble")
        leaf_key="probability" if kind=="forest" else "value"
        if kind=="boosting" and not finite(predictor["baseline"]):raise ValueError("Invalid boosting baseline")
        for tree in predictor["trees"]:
            n=len(tree["left"])
            if not n or any(len(tree[k])!=n for k in ("right","feature","threshold",leaf_key)):
                raise ValueError("Inconsistent tree arrays")
            for i in range(n):
                left,right=tree["left"][i],tree["right"][i]
                if not finite(tree[leaf_key][i]):raise ValueError("Invalid tree numeric value")
                if kind=="forest" and not 0<=tree[leaf_key][i]<=1:raise ValueError("Invalid leaf probability")
                if left==-1 and right==-1:continue
                if not finite(tree["threshold"][i]):raise ValueError("Invalid split threshold")
                if type(left) is not int or type(right) is not int or not (i<left<n and i<right<n):
                    raise ValueError("Tree children must be later nodes; cycle or invalid node")
                if type(tree["feature"][i]) is not int or not 0<=tree["feature"][i]<dimension:
                    raise ValueError("Invalid tree feature")
    elif kind=="ensemble" and depth==0:
        if not predictor["members"]:raise ValueError("Empty ensemble")
        for member in predictor["members"]:
            if member["kind"] not in ("logistic","forest","boosting"):raise ValueError("Unsupported ensemble member")
            validate_predictor(member,dimension,depth+1)
    else:raise ValueError("Unsupported predictor kind")


def _forest_probability(x,predictor):
    # sklearn's tree predictor casts input values to IEEE float32.
    x=[struct.unpack("f",struct.pack("f",v))[0] for v in x]
    total=0.0
    for tree in predictor["trees"]:
        node=0
        while tree["left"][node] >= 0:
            node=tree["left"][node] if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
        total += tree["probability"][node]
    return total/len(predictor["trees"])


def _boosting_logit(x,predictor):
    # HistGradientBoosting compares float64 inputs with float64 split thresholds; leaves are additive log-odds.
    total=predictor["baseline"]
    for tree in predictor["trees"]:
        node=0
        while tree["left"][node] >= 0:
            node=tree["left"][node] if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
        total += tree["value"][node]
    return total


def _logit(probability):
    probability=max(1e-6,min(1-1e-6,probability))
    return math.log(probability/(1-probability))


def _member_probability(x,predictor):
    kind=predictor["kind"]
    if kind=="logistic":
        return sigmoid(predictor["intercept"]+sum(w*(v-m)/s for v,w,m,s in zip(x,predictor["coef"],predictor["center"],predictor["scale"])))
    if kind=="forest":
        return _forest_probability(x,predictor)
    if kind=="boosting":
        return sigmoid(_boosting_logit(x,predictor))
    raise ValueError("Unsupported ensemble member")


def raw_score(features, model):
    """Uncalibrated score fed to the Platt calibrator (a logit for probabilistic predictors)."""
    x=[features[i] for i in model["feature_indices"]]
    predictor=model["predictor"]
    kind=predictor["kind"]
    if kind == "robust":
        return max((v-m)/s for v,m,s in zip(x,predictor["center"],predictor["scale"]))
    if kind == "logistic":
        return predictor["intercept"]+sum(w*(v-m)/s for v,w,m,s in zip(
            x,predictor["coef"],predictor["center"],predictor["scale"]))
    if kind == "forest":
        return _logit(_forest_probability(x,predictor))
    if kind == "boosting":
        return _boosting_logit(x,predictor)
    if kind == "ensemble":
        return _logit(sum(_member_probability(x,m) for m in predictor["members"])/len(predictor["members"]))
    raise ValueError("Unsupported predictor")


def predict_features(features, model):
    if len(features) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in features):
        raise ValueError("Invalid feature vector")
    calibration=model["calibration"]
    return sigmoid(calibration["intercept"]+calibration["coef"]*raw_score(features,model))


# ------------------------------------------------------------------ alarm evidence (TreeSHAP)
# Path-dependent TreeSHAP (Lundberg et al., 2020) on the exported trees, standard library only.
# Same values as shap.TreeExplainer on the sklearn model (tests/test_shap.py). A path element is
# [feature, zero_fraction, one_fraction, permutation_weight]; index 0 is the root placeholder.
def _extend_path(path,zero,one,feature):
    depth=len(path)
    path.append([feature,zero,one,1.0 if depth==0 else 0.0])
    for i in range(depth-1,-1,-1):
        path[i+1][3]+=one*path[i][3]*(i+1)/(depth+1)
        path[i][3]=zero*path[i][3]*(depth-i)/(depth+1)


def _unwind_path(path,index):
    depth=len(path)-1
    zero,one=path[index][1],path[index][2]
    following=path[depth][3]
    for i in range(depth-1,-1,-1):
        if one!=0:
            weight=path[i][3]
            path[i][3]=following*(depth+1)/((i+1)*one)
            following=weight-path[i][3]*zero*(depth-i)/(depth+1)
        else:
            path[i][3]=path[i][3]*(depth+1)/(zero*(depth-i))
    for i in range(index,depth):
        path[i][:3]=path[i+1][:3]
    path.pop()


def _unwound_path_sum(path,index):
    depth=len(path)-1
    zero,one=path[index][1],path[index][2]
    following,total=path[depth][3],0.0
    for i in range(depth-1,-1,-1):
        if one!=0:
            part=following*(depth+1)/((i+1)*one)
            total+=part
            following=path[i][3]-part*zero*(depth-i)/(depth+1)
        elif zero!=0:
            total+=(path[i][3]/zero)/((depth-i)/(depth+1))
    return total


def _tree_shap(x,tree,phi):
    left,right,feature,threshold=tree["left"],tree["right"],tree["feature"],tree["threshold"]
    value,cover=tree["value"],tree["cover"]

    def recurse(node,parent,zero,one,split_feature):
        path=[element[:] for element in parent]
        _extend_path(path,zero,one,split_feature)
        if left[node]<0:
            for i in range(1,len(path)):
                phi[path[i][0]]+=_unwound_path_sum(path,i)*(path[i][2]-path[i][1])*value[node]
            return
        split=feature[node]
        hot,cold=(left[node],right[node]) if x[split]<=threshold[node] else (right[node],left[node])
        incoming_zero=incoming_one=1.0
        for k in range(len(path)):
            if path[k][0]==split:
                incoming_zero,incoming_one=path[k][1],path[k][2]
                _unwind_path(path,k)
                break
        recurse(hot,path,cover[hot]/cover[node]*incoming_zero,incoming_one,split)
        recurse(cold,path,cover[cold]/cover[node]*incoming_zero,0.0,split)
    recurse(0,[],1.0,1.0,-1)


def feature_contributions(features, model):
    """Per-input SHAP contributions to the uncalibrated log-odds (boosting or logistic predictor).

    Returns [(feature_name, contribution), ...] in model input order, or None for other predictors.
    Contributions plus the expected value sum to raw_score(features, model)."""
    x=[features[i] for i in model["feature_indices"]]
    predictor=model["predictor"]
    if predictor["kind"]=="logistic":
        phi=[w*(v-m)/s for v,w,m,s in zip(x,predictor["coef"],predictor["center"],predictor["scale"])]
    elif predictor["kind"]=="boosting":
        phi=[0.0]*len(x)
        for tree in predictor["trees"]:
            _tree_shap(x,tree,phi)
    else:
        return None
    return [(FEATURE_NAMES[i],v) for i,v in zip(model["feature_indices"],phi)]


def expected_raw_score(model):
    """Expected value of the TreeSHAP decomposition (training-cover-weighted mean leaf value)."""
    predictor=model["predictor"]
    if predictor["kind"]=="logistic":
        return predictor["intercept"]
    if predictor["kind"]!="boosting":
        return None
    total=predictor["baseline"]
    for tree in predictor["trees"]:
        total+=sum(v*c for v,c,l in zip(tree["value"],tree["cover"],tree["left"]) if l<0)/tree["cover"][0]
    return total


def mode_threshold(model, mode="standard"):
    """Decision threshold of an operating mode. Unknown modes are rejected."""
    if mode not in OPERATING_MODES:
        raise ValueError("알 수 없는 운영 모드입니다: "+str(mode))
    if mode=="standard":
        return float(model["threshold"])
    modes=model.get("operating_modes")
    if not modes:
        raise ValueError("이 모델에는 운영 모드 임계값이 없습니다. 표준 모드를 사용하세요.")
    return float(modes[mode]["threshold"])


class StreamAnalyzer:
    def __init__(self, model, mode="standard"):
        self.model=model
        self.mode=mode
        self.threshold=mode_threshold(model,mode)
        self.extractor=CausalFeatureExtractor(model["max_window"],model["max_gap_seconds"])
        self.votes=deque(maxlen=model["alarm_policy"]["window"])
        self.previous_alarm=False
        self.previous_values=None
        self.previous_timestamp=None
        self.blocked_constant_channels={}

    def invalid_record(self,timestamp,reason):
        self.extractor.reset()
        self.votes.clear()
        self.previous_alarm=False
        self.previous_values=None
        self.previous_timestamp=None
        self.blocked_constant_channels={}
        return {"timestamp":timestamp,"score":None,"predicted":False,"alarm":False,
                "alarm_event":False,"status":"INVALID","reason":reason,"ood_review":False,
                "n_history":0,"gap_reset":True,"duplicate":False}

    def update(self, values, timestamp):
        try:
            parsed=parse_timestamp(timestamp)
            values=tuple(float(v) for v in values)
            if len(values)!=3 or not all(math.isfinite(v) for v in values):
                raise ValueError("센서값에 결측·무한값·숫자 오류가 있습니다.")
            still_constant={i:v for i,v in self.blocked_constant_channels.items() if values[i]==v}
            self.blocked_constant_channels=still_constant
            if still_constant:
                self.extractor.reset();self.votes.clear();self.previous_alarm=False
                self.previous_timestamp=parsed;self.previous_values=values
                record={"timestamp":str(timestamp),"score":None,"predicted":False,"alarm":False,
                        "alarm_event":False,"status":"REVIEW","reason":"센서 고정 상태 지속: 값이 변할 때까지 판정 보류",
                        "ood_review":False,"n_history":0,"gap_reset":True,"duplicate":False}
                record.update(dict(zip(CHANNELS,values)));return record
            if self.previous_timestamp is not None:
                delta=(parsed-self.previous_timestamp).total_seconds()
                if delta==0 and values==self.previous_values:
                    return {"timestamp":str(timestamp),"score":None,"predicted":False,"alarm":False,
                            "alarm_event":False,"status":"REVIEW","reason":"동일 시각·센서 중복행: 점수에서 제외",
                            "ood_review":False,"n_history":len(self.extractor.history),"gap_reset":False,"duplicate":True}
                if delta<=0:
                    return self.invalid_record(str(timestamp),"시각이 증가하지 않습니다. 구간을 초기화했습니다.")
                if abs(delta-self.model["expected_sample_interval_seconds"])>0.015 and delta<=self.model["max_gap_seconds"]:
                    return self.invalid_record(str(timestamp),"0.1초 표본 간격과 맞지 않습니다. 현장 재보정 필요")
            features=self.extractor.update(values,parsed)
        except (ValueError,TypeError,OverflowError) as exc:
            return self.invalid_record(str(timestamp),str(exc))
        self.previous_timestamp=parsed
        self.previous_values=values
        if features["gap_reset"]:
            self.votes.clear()
            self.previous_alarm=False
        score=predict_features(features["features"],self.model)
        warmed=features["n_history"]>=self.model["alarm_policy"]["minimum_history"]
        # v1.2: a row is judged only after >= minimum_history samples (n=1 is an all-zero vector,
        # n=2 a correlation of exactly +/-1). Unjudged rows neither predict nor vote.
        predicted=bool(warmed and score>=self.threshold)
        ood=any(abs(v)>lim for v,lim in zip(values,self.model["review_absolute_limits"]))
        frozen=features["frozen_channels"]
        if warmed:
            self.votes.append(int(predicted))
        alarm=warmed and sum(self.votes)>=self.model["alarm_policy"]["votes_required"] and not ood and not frozen
        status="ALARM" if alarm else ("WATCH" if predicted else "NORMAL")
        if self.mode=="screening":
            reason="점검 후보: 선별 기준 지속 초과" if alarm else ("선별 기준 일시 초과: 점검 후보 관찰" if predicted else "선별 기준 미만")
        else:
            reason="모델 이상점수 및 지속성 조건 충족" if alarm else ("일시 이상점수: 지속 여부 관찰" if predicted else "데이터셋 기준 정상 범위")
        if not warmed:
            status="WARMUP";reason="현재 구간의 표본 수집 중"
        if features["gap_reset"]:
            reason+=" / 시간 공백으로 이력 초기화"
        if ood or frozen:
            status="REVIEW";alarm=False
            reason="학습 범위를 크게 벗어난 값: 센서 단위·배선·조건 확인" if ood else "센서값 고정 감지: "+", ".join(frozen)
            self.votes.clear();self.previous_alarm=False;self.extractor.reset()
            if frozen:
                self.blocked_constant_channels={CHANNELS.index(c):values[CHANNELS.index(c)] for c in frozen}
        observations=[]
        reference=self.model.get("normal_reference",{}).get("feature_p99",{})
        for name,label in EVIDENCE_FEATURES:
            upper=reference.get(name)
            value=features["features"][FEATURE_NAMES.index(name)]
            if upper is not None and upper>0 and value>upper:
                observations.append((value/upper,f"{label} {value:.3g} > 정상학습 p99 {upper:.3g}"))
        observations=[s for _,s in sorted(observations,reverse=True)[:2]]
        if observations:reason+=" / 정상 기준 대비 관측: "+"; ".join(observations)
        event=bool(alarm and not self.previous_alarm)
        self.previous_alarm=bool(alarm)
        # Evidence below is text only: score, decision, vote and alarm are already fixed.
        lower_still=False
        contributions=[]
        if status in ("WATCH","ALARM"):
            lower_std=features["features"][FEATURE_NAMES.index("AI1_Vibration_std")]
            if lower_std<=LOWER_VIBRATION_STILL_STD:
                lower_still=True
                reason+=(f" / 하부 진동 정지 상태(AI1 진동 표준편차 {lower_std:.3g} ≤ {LOWER_VIBRATION_STILL_STD:.3g}):"
                         " 오경보가 집중된 조건이므로 하부 센서 부착·배선·커넥터 확인 후 재판정")
        if event:
            # SHAP costs far more than scoring, so it is computed only when an alarm is entered.
            contributions=sorted(((v,n) for n,v in feature_contributions(features["features"],self.model) or [] if v>0),reverse=True)[:3]
            contributions=[[FEATURE_LABELS.get(n,n),round(v,3)] for v,n in contributions]
            if contributions:
                reason+=" / 주요 기여(SHAP, 경보 진입 시점): "+", ".join(f"{n} +{v:.2f}" for n,v in contributions[:2])
        record={"timestamp":str(timestamp),"score":score,"predicted":bool(predicted),"alarm":bool(alarm),
                "alarm_event":event,"status":status,"reason":reason,"ood_review":bool(ood),
                "n_history":features["n_history"],"gap_reset":features["gap_reset"],"duplicate":False}
        record["observations"]=observations
        record["lower_sensor_check"]=lower_still
        record["contributions"]=contributions
        record.update(dict(zip(CHANNELS,values)))
        return record


def rank_segments(records):
    """Inspection priority of continuous segments (history restarts), the rule of analysis.py:
    share of judged samples at/above the active threshold, ties broken by mean score.
    Adds record["segment"] (1-based) and returns judged segments ordered by priority."""
    segment,groups=0,{}
    for i,r in enumerate(records):
        if i==0 or r["gap_reset"] or (r["score"] is not None and r["n_history"]==1):
            segment+=1
        r["segment"]=segment
        if r["status"] in ("NORMAL","WATCH","ALARM"):
            groups.setdefault(segment,[]).append(r)
    ranked=[]
    for seg,rows in groups.items():
        ranked.append({"segment":seg,"start":rows[0]["timestamp"],"end":rows[-1]["timestamp"],
                       "first_csv_line":rows[0].get("csv_line"),"judged_rows":len(rows),
                       "flag_ratio":sum(r["predicted"] for r in rows)/len(rows),
                       "mean_score":sum(r["score"] for r in rows)/len(rows),
                       "alarm_events":sum(r["alarm_event"] for r in rows)})
    ranked.sort(key=lambda s:(-s["flag_ratio"],-s["mean_score"]))
    for rank,s in enumerate(ranked,1):
        s["rank"]=rank
    return ranked


def analyze_csv(path,model,mode="standard"):
    started=time.perf_counter()
    if isinstance(model,(str,Path)):
        model=load_model(model)
    analyzer=StreamAnalyzer(model,mode)
    records=[]
    with Path(path).open("r",encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if len(reader.fieldnames or [])!=len(set(reader.fieldnames or [])):
            raise ValueError("중복 CSV 열 이름이 있습니다.")
        required=["TimeStamp",*CHANNELS]
        missing=[c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("필수 열 누락: "+", ".join(missing))
        for number,row in enumerate(reader,2):
            if None in row:
                raise ValueError(f"CSV {number}행의 열 수가 맞지 않습니다.")
            rec=analyzer.update([row.get(c) for c in CHANNELS],row.get("TimeStamp",""))
            rec["csv_line"]=number
            records.append(rec)
    if not records:raise ValueError("데이터 행이 없는 파일입니다.")
    counts={status:sum(r["status"]==status for r in records) for status in STATUSES}
    summary={"rows":len(records),"scored_rows":sum(r["score"] is not None for r in records),
             "anomaly_rows":sum(r["predicted"] for r in records),"alarm_rows":sum(r["alarm"] for r in records),
             "alarm_events":sum(r["alarm_event"] for r in records),"review_rows":counts["REVIEW"],
             "normal_rows":counts["NORMAL"],"invalid_rows":counts["INVALID"],"warmup_rows":counts["WARMUP"],
             "duplicate_rows":sum(r["duplicate"] for r in records),"status_counts":counts,
             "elapsed_seconds":time.perf_counter()-started}
    judged=[r for r in records if r["status"] in ("NORMAL","WATCH","ALARM")]
    # Share of fully warmed samples above the active threshold. In screening mode this
    # ranks files/segments for inspection (validated as a ranking, not as an alarm).
    summary["flag_ratio"]=(sum(r["predicted"] for r in judged)/len(judged)) if judged else None
    segments=rank_segments(records)
    summary["segments"]=max((r["segment"] for r in records),default=0)
    summary["ranked_segments"]=len(segments)
    return {"records":records,"summary":summary,"segments":segments,"metadata":{"model_id":model["model_id"],
            "mode":"SHADOW_CSV_REPLAY","expected_sampling_hz":model["expected_sampling_hz"],
            "calibration_scope":"KAMP internal split; site validation required",
            "threshold":analyzer.threshold,"operating_mode":mode,"operating_mode_label":MODE_LABELS[mode],
            "source_file":str(Path(path).resolve())}}
