"""Train 30-day readmission model. Leakage-safe, group-by-patient split."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "diabetic_data.csv"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
EXCLUDE_DISCHARGE = [11, 13, 14, 19, 20, 21]  # expired / hospice: cannot be readmitted
MED_COLS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone",
    "acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin",
    "glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"]

def icd_group(code):
    if pd.isna(code): return "Missing"
    s = str(code).strip()
    if s.startswith("V") or s.startswith("E"): return "Supplementary/Injury-ext"
    try: v = float(s.split(".")[0])
    except: return "Other"
    if 390 <= v <= 459 or v == 785: return "Circulatory"
    if 460 <= v <= 519 or v == 786: return "Respiratory"
    if 520 <= v <= 579 or v == 787: return "Digestive"
    if 250 <= v < 251: return "Diabetes"
    if 800 <= v <= 999: return "Injury"
    if 710 <= v <= 739: return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "Genitourinary"
    if 140 <= v <= 239: return "Neoplasms"
    if 240 <= v <= 279: return "Endocrine/Metabolic"
    return "Other"

def engineer(df):
    df = df.copy()
    for c in ["diag_1","diag_2","diag_3"]:
        df[c+"_grp"] = df[c].apply(icd_group)
    df["total_prior_visits"] = df["number_outpatient"]+df["number_emergency"]+df["number_inpatient"]
    df["had_prior_inpatient"] = (df["number_inpatient"]>0).astype(int)
    df["polypharmacy"] = (df["num_medications"]>8).astype(int)
    df["long_stay"] = (df["time_in_hospital"]>4).astype(int)
    df["num_med_changes"] = sum((df[c].isin(["Up","Down"])).astype(int) for c in MED_COLS)
    df["insulin_changed"] = df["insulin"].isin(["Up","Down"]).astype(int)
    return df

def main():
    df = pd.read_csv(DATA, na_values=["?"], low_memory=False)
    n0 = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE)].copy()
    print(f"rows {n0} -> {len(df)} after excluding expired/hospice ({n0-len(df)})")
    df["y"] = (df["readmitted"]=="<30").astype(int)
    print("target rate:", df["y"].mean().round(4), df["y"].value_counts().to_dict())
    df = engineer(df)

    drop_cols = ["encounter_id","patient_nbr","readmitted","y","weight","payer_code",
                 "diag_1","diag_2","diag_3","examide","citoglipton"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["y"].values
    groups = df["patient_nbr"].values

    # group-aware 80/20 split via StratifiedGroupKFold (fold 0 = test)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y[tr_idx], y[te_idx]
    assert len(set(groups[tr_idx]) & set(groups[te_idx]))==0, "patient leakage!"
    print(f"train {Xtr.shape} pos_rate={ytr.mean():.4f} | test {Xte.shape} pos_rate={yte.mean():.4f}")

    # Top-N grouping fitted on TRAIN only
    top_spec = Xtr["medical_specialty"].value_counts().head(8).index.tolist()
    Xtr["med_spec_grp"] = Xtr["medical_specialty"].where(Xtr["medical_specialty"].isin(top_spec), "Other").fillna("Missing")
    Xte["med_spec_grp"] = Xte["medical_specialty"].where(Xte["medical_specialty"].isin(top_spec), "Other").fillna("Missing")

    num_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
        "total_prior_visits","num_med_changes"]
    cat_cols = ["race","gender","age","max_glu_serum","A1Cresult","change","diabetesMed",
        "med_spec_grp","diag_1_grp","diag_2_grp","diag_3_grp",
        "admission_type_id","discharge_disposition_id","admission_source_id",
        "had_prior_inpatient","polypharmacy","long_stay","insulin_changed"]
    # keep only existing
    num_cols=[c for c in num_cols if c in Xtr.columns]
    cat_cols=[c for c in cat_cols if c in Xtr.columns]
    feat_cols = num_cols+cat_cols+MED_COLS
    # med cols: fill NA + keep as categorical (only cols still present)
    med_exist=[c for c in MED_COLS if c in Xtr.columns]
    feat_cols = num_cols+cat_cols+med_exist
    for c in med_exist:
        Xtr[c]=Xtr[c].fillna("No"); Xte[c]=Xte[c].fillna("No")

    def make_pre(sparse=True):
        return ColumnTransformer([
            ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                ("oh",OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))]), cat_cols+med_exist),
        ])
    pre = make_pre(sparse=True)
    pre_dense = make_pre(sparse=False)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gtr = groups[tr_idx]
    models = {
        "logreg": (Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
                   {"clf__C":[0.1,1.0,5.0]}),
        "rf": (Pipeline([("pre",pre),("clf",RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))]),
               {"clf__n_estimators":[200],"clf__max_depth":[14,None],"clf__min_samples_leaf":[2]}),
        "hgb": (Pipeline([("pre",pre_dense),("clf",HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
                {"clf__learning_rate":[0.07],"clf__max_depth":[7,11],"clf__max_iter":[200]}),
    }
    results, fitted = {}, {}
    # baseline
    dummy = Pipeline([("pre",pre),("clf",DummyClassifier(strategy="most_frequent"))])
    dummy.fit(Xtr[feat_cols], ytr)
    fitted["baseline"]=dummy

    for name,(pipe,grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv.split(Xtr[feat_cols],ytr,gtr),
                          scoring="recall", n_jobs=-1, verbose=0)
        gs.fit(Xtr[feat_cols], ytr)
        fitted[name]=gs.best_estimator_
        print(f"{name}: best={gs.best_params_} cv_recall={gs.best_score_:.3f}")
        results[name]={"best_params":str(gs.best_params_),"cv_recall":round(float(gs.best_score_),4)}

    # test evaluation
    summary={}
    for name,pipe in fitted.items():
        p = pipe.predict(Xte[feat_cols]); s = pipe.predict_proba(Xte[feat_cols])[:,1]
        summary[name]=dict(
            precision=round(float(precision_score(yte,p,zero_division=0)),4),
            recall=round(float(recall_score(yte,p)),4),
            f1=round(float(f1_score(yte,p)),4),
            roc_auc=round(float(roc_auc_score(yte,s)),4),
            pr_auc=round(float(average_precision_score(yte,s)),4),
            tn=int(confusion_matrix(yte,p)[0,0]), fp=int(confusion_matrix(yte,p)[0,1]),
            fn=int(confusion_matrix(yte,p)[1,0]), tp=int(confusion_matrix(yte,p)[1,1]),
        )
        print(name, summary[name])
    summary["cv"]=results
    summary["n_train"]=int(len(Xtr)); summary["n_test"]=int(len(Xte))
    summary["pos_rate_test"]=round(float(yte.mean()),4)
    summary["top_specialties"]=top_spec
    summary["feature_cols"]=feat_cols
    summary["threshold"]=0.5
    with open(MODELS_DIR / "metrics.json","w") as f: json.dump(summary,f,indent=2)

    # pick best by recall then PR-AUC (excluding baseline)
    best = max(["logreg","rf","hgb"], key=lambda k:(summary[k]["recall"], summary[k]["pr_auc"]))
    print("BEST:", best)
    joblib.dump({"pipeline":fitted[best],"model_name":best,"top_specialties":top_spec,
        "feature_cols":feat_cols,"threshold":0.5}, MODELS_DIR / "model.joblib")
    print("saved model.joblib + metrics.json")

    # charts: ROC + confusion matrix of best
    fig,ax=plt.subplots(figsize=(6,5))
    for name,pipe in fitted.items():
        s=pipe.predict_proba(Xte[feat_cols])[:,1]
        RocCurveDisplay.from_predictions(yte,s,name=name,ax=ax)
    ax.plot([0,1],[0,1],"k--",lw=1); ax.set_title("ROC — test set")
    fig.tight_layout(); fig.savefig(ASSETS_DIR / "roc.png",dpi=120); plt.close(fig)
    cm=np.array([[summary[best]["tn"],summary[best]["fp"]],[summary[best]["fn"],summary[best]["tp"]]])
    fig,ax=plt.subplots(figsize=(4,4))
    ax.imshow(cm); ax.set_title(f"Confusion matrix — {best} (test)")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["pred 0","pred 1"]); ax.set_yticklabels(["true 0","true 1"])
    for i in range(2):
        for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=14,color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(ASSETS_DIR / "confusion.png",dpi=120); plt.close(fig)

if __name__=="__main__":
    main()
