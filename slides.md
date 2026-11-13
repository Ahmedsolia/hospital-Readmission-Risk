# Presentation — 10 min, 10–12 slides (outline; build in Canva/PowerPoint)

1. Title: 30-Day Readmission Risk for Diabetic Patients + live app link + disclaimer (not a diagnosis)
2. Problem: costly 30-day returns; who gets scarce follow-up? Decision at discharge.
3. Data: 101,766 encounters / 71,518 patients, 50 cols; target <30 = 11.4%; excluded 2,423 expired/hospice; `?` missing, ICD-9 codes, Top-8 specialties.
4. EDA insight 1: prior inpatient stays sharply raise risk (chart).
5. EDA insight 2: A1C=High + med/insulin changes + emergency admission raise risk; age 70–80 peak volume.
6. Pipeline (no leakage): group-by-patient split, ColumnTransformer in Pipeline, ICD-9 groups, med-change counts; train-only fitting.
7. Models: baseline (recall 0) + LogReg / RF / HGB, 5-fold GroupStratified CV scoring recall.
8. Results table (test): HGB recall 0.557, ROC-AUC 0.666, PR-AUC 0.212; confusion matrix TN 11881 FP 5726 FN 1003 TP 1260; ROC chart.
9. Error/fairness: false alarms vs misses trade-off; check recall by race/gender/age before deployment.
10. Recommendation: HIGH ≥0.50 → 48h call + 7-day slot + pharmacist review; MEDIUM 0.20–0.50 → 1-week call; LOW → standard. Limitations: 1999–2008, diabetes-only, recalibrate.
11. Demo: discharge form → risk + factors + checklist (30–60s video for LinkedIn).
12. Next steps: pilot A/B, add adherence data, threshold tuning by cost, yearly recalibration. Links: app, repo, dataset https://archive.ics.uci.edu/dataset/296

# LinkedIn post (draft — replace [link] after deploy)

Hospitals only find out about readmissions after they happen. In my NextGen ML project, I built a discharge tool that flags diabetic patients at risk of returning within 30 days — before they leave.

- Trained LogReg / Random Forest / HistGradientBoosting on 99,343 real hospital encounters (group-by-patient split, no leakage)
- Best model catches 55.7% of 30-day returns (ROC-AUC 0.67) — baseline catches 0%
- Biggest lesson: accuracy (≈82% for a dummy!) lies under 11% imbalance — recall + PR-AUC is what the care team needs; every false alarm has a cost too

Try the live app: [link] · Code: [GitHub link]
Dataset: UCI Diabetes 130-US Hospitals 1999–2008. Screening aid only — not a diagnosis.
#MachineLearning #DataScience #HealthcareAI
