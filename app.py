"""Hospital Readmission Risk — Streamlit discharge tool."""
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
BUNDLE = joblib.load(MODEL_PATH)
PIPE = BUNDLE["pipeline"]
TOP_SPEC = BUNDLE["top_specialties"]
FEATS = BUNDLE["feature_cols"]
THRESHOLD = float(BUNDLE.get("threshold", 0.5))

ADMISSION_TYPES = {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",5:"Not Available",6:"NULL",7:"Trauma Center",8:"Not Mapped"}
DISCHARGE = {1:"Discharged to home",2:"Transfer to short-term hospital",3:"Transfer to SNF",4:"Transfer to ICF",
 5:"Other inpatient care",6:"Home with home health",7:"Left AMA",8:"Home with IV care",9:"Admitted as inpatient",
 10:"Neonate to another hospital",11:"Expired",12:"Outpatient services expected",13:"Hospice/home",14:"Hospice/medical facility",
 15:"Swing bed",16:"Referred outpatient (other)",17:"Referred outpatient (this)",18:"NULL",19:"Expired at home (hospice)",
 20:"Expired in facility (hospice)",22:"Rehab facility",23:"Long-term care hospital",24:"Nursing facility (Medicaid)",
 25:"Not Mapped",27:"Federal facility",28:"Psychiatric unit",29:"Critical Access Hospital"}
SOURCE = {1:"Physician Referral",2:"Clinic Referral",3:"HMO Referral",4:"Transfer from hospital",5:"Transfer from SNF",
 6:"Transfer from other facility",7:"Emergency Room",8:"Court/Law",9:"Not Available",10:"Transfer from critical-access",
 11:"Normal Delivery",12:"Premature Delivery",13:"Sick Baby",14:"Extramural Birth",17:"NULL",18:"Other home-health agency",
 19:"Readmission to same home-health",20:"Not Mapped",22:"Transfer hospital/same-facility",25:"Ambulatory surgery",26:"Transfer from hospice"}
BLOCKED_DISCHARGE = [11,13,14,19,20,21]
DIAG_GROUPS = ["Circulatory","Diabetes","Respiratory","Digestive","Endocrine/Metabolic","Genitourinary",
 "Injury","Musculoskeletal","Neoplasms","Supplementary/Injury-ext","Other","Missing"]

st.set_page_config(page_title="Discharge Readmission Risk", layout="centered")
st.title("🏥 30-Day Readmission Risk — Discharge Tool")
st.caption("Operations aid for the care team. **Not a medical diagnosis.** Model: HistGradientBoosting trained on 99,343 encounters (Diabetes 130-US Hospitals).")
st.info("How to use: fill the discharge form → get risk + tier + follow-up checklist.", icon="ℹ️")

with st.form("discharge"):
    c1,c2 = st.columns(2)
    with c1:
        age = st.selectbox("Age band",["[50-60)","[60-70)","[70-80)","[80-90)","[40-50)","[30-40)","[20-30)","[10-20)","[0-10)","[90-100)"])
        gender = st.selectbox("Gender",["Female","Male"])
        race = st.selectbox("Race",["Caucasian","AfricanAmerican","Hispanic","Asian","Other"])
        specialty = st.selectbox("Medical specialty",["InternalMedicine","Emergency/Trauma","Family/GeneralPractice","Cardiology","Surgery-General","Nephrology","Orthopedics","Orthopedics-Reconstructive","Other/Missing"])
        adm_type = st.selectbox("Admission type", list(ADMISSION_TYPES.items()), format_func=lambda x:f"{x[0]} — {x[1]}")
        adm_src = st.selectbox("Admission source", list(SOURCE.items()), format_func=lambda x:f"{x[0]} — {x[1]}")
        disch = st.selectbox("Discharge disposition", list(DISCHARGE.items()), format_func=lambda x:f"{x[0]} — {x[1]}")
    with c2:
        tih = st.number_input("Days in hospital",0,30,4)
        n_lab = st.number_input("Lab procedures",0,200,45)
        n_proc = st.number_input("Other procedures",0,20,1)
        n_med = st.number_input("Medications count",0,100,15)
        n_diag = st.number_input("Diagnoses count",1,16,8)
        n_out = st.number_input("Prior outpatient visits",0,20,0)
        n_em = st.number_input("Prior emergency visits",0,20,0)
        n_inp = st.number_input("Prior inpatient stays",0,20,0)
    c3,c4 = st.columns(2)
    with c3:
        d1 = st.selectbox("Primary diagnosis group",DIAG_GROUPS,index=1)
        d2 = st.selectbox("Secondary diagnosis group",DIAG_GROUPS,index=9)
        d3 = st.selectbox("Tertiary diagnosis group",DIAG_GROUPS,index=11)
        a1c = st.selectbox("A1C result",["None","Norm","High"])
        glu = st.selectbox("Max glucose serum",["None","Norm","High"])
    with c4:
        change = st.selectbox("Med change during stay",["No","Ch"])
        diab_med = st.selectbox("Diabetes medication",["Yes","No"])
        insulin = st.selectbox("Insulin",["Steady","No","Up","Down"])
        metformin = st.selectbox("Metformin",["No","Steady","Up","Down"])
        glipizide = st.selectbox("Glipizide",["No","Steady","Up","Down"])
        glyburide = st.selectbox("Glyburide",["No","Steady","Up","Down"])
    threshold = st.slider("Alert threshold (advanced)",0.1,0.8,0.5,0.05)
    submitted = st.form_submit_button("Predict 30-day risk")

def build_row():
    spec = specialty if specialty in TOP_SPEC else ("Missing" if specialty=="Other/Missing" else "Other")
    med_change_map = {"metformin":metformin,"insulin":insulin,"glipizide":glipizide,"glyburide":glyburide}
    n_changes = sum(1 for v in med_change_map.values() if v in ("Up","Down"))
    total_prior = n_out+n_em+n_inp
    row = {
        "time_in_hospital":tih,"num_lab_procedures":n_lab,"num_procedures":n_proc,"num_medications":n_med,
        "number_outpatient":n_out,"number_emergency":n_em,"number_inpatient":n_inp,"number_diagnoses":n_diag,
        "total_prior_visits":total_prior,"num_med_changes":n_changes,
        "race":race,"gender":gender,"age":age,"max_glu_serum":glu,"A1Cresult":a1c,"change":change,
        "diabetesMed":diab_med,"med_spec_grp":spec,"diag_1_grp":d1,"diag_2_grp":d2,"diag_3_grp":d3,
        "admission_type_id":adm_type[0],"discharge_disposition_id":disch[0],"admission_source_id":adm_src[0],
        "had_prior_inpatient":int(n_inp>0),"polypharmacy":int(n_med>8),"long_stay":int(tih>4),
        "insulin_changed":int(insulin in ("Up","Down")),
        "metformin":metformin,"repaglinide":"No","nateglinide":"No","chlorpropamide":"No","glimepiride":"No",
        "acetohexamide":"No","glipizide":glipizide,"glyburide":glyburide,"tolbutamide":"No","pioglitazone":"No",
        "rosiglitazone":"No","acarbose":"No","miglitol":"No","troglitazone":"No","tolazamide":"No",
        "insulin":insulin,"glyburide-metformin":"No","glipizide-metformin":"No","glimepiride-pioglitazone":"No",
        "metformin-rosiglitazone":"No","metformin-pioglitazone":"No",
    }
    return pd.DataFrame([row])[FEATS], n_changes

if submitted:
    if disch[0] in BLOCKED_DISCHARGE:
        st.error("This discharge outcome (expired/hospice) cannot be scored for readmission. No prediction made.")
    else:
        X,_ = build_row()
        try:
            proba = float(PIPE.predict_proba(X)[0,1])
        except Exception as e:
            st.error(f"Prediction failed without crashing: {e}")
            st.stop()
        tier = "🔴 HIGH" if proba>=threshold else ("🟡 MEDIUM" if proba>=0.2 else "🟢 LOW")
        st.subheader(f"{tier} — {proba:.1%} 30-day readmission risk")
        st.progress(min(proba,1.0))
        factors=[]
        if n_inp>0: factors.append(f"Prior inpatient stays: {n_inp} (strongest signal in EDA)")
        if n_med>8: factors.append(f"Polypharmacy: {n_med} meds")
        if tih>4: factors.append(f"Long stay: {tih} days")
        if insulin in ("Up","Down"): factors.append("Insulin dose changed during stay")
        if a1c=="High": factors.append("A1C = High (poor glycemic control)")
        if adm_type[0]==1: factors.append("Emergency admission")
        if change=="Ch": factors.append("Medication regimen changed")
        st.write("**Top factors raising risk:**")
        st.write("\n".join(f"- {f}" for f in factors) if factors else "- No major flags; baseline population risk ~11%.")
        st.write("**Recommended follow-up:**")
        if proba>=threshold:
            st.write("- 48-hour nurse phone call + 7-day clinic slot + pharmacist med review + home-health evaluation")
        elif proba>=0.2:
            st.write("- 1-week nurse phone call + confirm 14-day follow-up appointment")
        else:
            st.write("- Standard discharge instructions + routine follow-up")
        st.caption(f"Operating threshold {threshold:.2f} · model threshold 0.50 · P(recall 0.56, precision 0.18 on test). Tune threshold to trade missed cases vs false alarms.")
