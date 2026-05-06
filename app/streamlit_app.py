import streamlit as st
import pandas as pd
import joblib

# ======================================
# LOAD MODEL
# ======================================
model = joblib.load("data/random_forest_model.pkl")

st.set_page_config(page_title="OvaPredict", layout="wide")

st.title("🌸 OvaPredict: PCOS Risk Analysis & Prediction")
st.markdown("Enter patient details to predict PCOS risk.")

# ======================================
# INPUT FIELDS
# ======================================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (yrs)", 10, 60, 25)
    weight = st.number_input("Weight (Kg)", 20.0, 150.0, 60.0)
    height = st.number_input("Height(Cm)", 100.0, 220.0, 165.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    blood_group = st.selectbox("Blood Group", [11, 12, 13, 14, 15, 16, 17, 18])
    pulse_rate = st.number_input("Pulse rate(bpm)", 40, 150, 72)
    rr = st.number_input("RR (breaths/min)", 10, 40, 18)
    hb = st.number_input("Hb(g/dl)", 5.0, 20.0, 12.5)
    cycle = st.selectbox("Cycle(R/I)", [1, 2])
    cycle_length = st.number_input("Cycle length(days)", 1, 90, 30)
    marriage_status = st.number_input("Marraige Status (Yrs)", 0, 30, 3)
    pregnant = st.selectbox("Pregnant(Y/N)", [0, 1])
    abortions = st.number_input("No. of aborptions", 0, 10, 1)

with col2:
    beta_hcg_1 = st.number_input("I beta-HCG(mIU/mL)", 0.0, 1000.0, 1.2)
    beta_hcg_2 = st.number_input("II beta-HCG(mIU/mL)", 0.0, 1000.0, 1.3)
    hip = st.number_input("Hip(inch)", 20.0, 80.0, 36.0)
    waist = st.number_input("Waist(inch)", 20.0, 80.0, 30.0)
    wh_ratio = st.number_input("Waist:Hip Ratio", 0.1, 2.0, 0.83)
    tsh = st.number_input("TSH (mIU/L)", 0.0, 20.0, 2.5)
    amh = st.number_input("AMH(ng/mL)", 0.0, 20.0, 4.2)
    prl = st.number_input("PRL(ng/mL)", 0.0, 100.0, 15.0)
    vit_d3 = st.number_input("Vit D3 (ng/mL)", 0.0, 100.0, 25.0)
    prg = st.number_input("PRG(ng/mL)", 0.0, 20.0, 1.1)
    rbs = st.number_input("RBS(mg/dl)", 50.0, 300.0, 90.0)
    weight_gain = st.selectbox("Weight gain(Y/N)", [0, 1])
    hair_growth = st.selectbox("hair_growth(Y/N)", [0, 1])

with col3:
    skin_darkening = st.selectbox("Skin darkening (Y/N)", [0, 1])
    hair_loss = st.selectbox("Hair loss(Y/N)", [0, 1])
    pimples = st.selectbox("Pimples(Y/N)", [0, 1])
    fast_food = st.selectbox("Fast food (Y/N)", [0, 1])
    exercise = st.selectbox("Reg.Exercise(Y/N)", [0, 1])
    bp_sys = st.number_input("BP _Systolic (mmHg)", 80, 200, 120)
    bp_dia = st.number_input("BP _Diastolic (mmHg)", 40, 150, 80)
    follicle_l = st.number_input("Follicle No. (L)", 0, 50, 8)
    follicle_r = st.number_input("Follicle No. (R)", 0, 50, 7)
    avg_f_l = st.number_input("Avg. F size (L) (mm)", 0.0, 50.0, 14.0)
    avg_f_r = st.number_input("Avg. F size (R) (mm)", 0.0, 50.0, 15.0)
    endometrium = st.number_input("Endometrium (mm)", 0.0, 30.0, 9.0)

# ======================================
# PREDICTION
# ======================================

if st.button("Predict PCOS Risk"):

    data = pd.DataFrame([[ 
        age, weight, height, bmi,
        blood_group, pulse_rate, rr,
        hb, cycle, cycle_length,
        marriage_status, pregnant,
        abortions, beta_hcg_1,
        beta_hcg_2, hip, waist,
        wh_ratio, tsh, amh,
        prl, vit_d3, prg,
        rbs, weight_gain, hair_growth,
        skin_darkening, hair_loss,
        pimples, fast_food,
        exercise, bp_sys,
        bp_dia, follicle_l,
        follicle_r, avg_f_l,
        avg_f_r, endometrium
    ]], columns=[
        'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI',
        'Blood Group', 'Pulse rate(bpm)', 'RR (breaths/min)',
        'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)',
        'Marraige Status (Yrs)', 'Pregnant(Y/N)',
        'No. of aborptions', 'I beta-HCG(mIU/mL)',
        'II beta-HCG(mIU/mL)', 'Hip(inch)', 'Waist(inch)',
        'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)',
        'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)',
        'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair_growth(Y/N)',
        'Skin darkening (Y/N)', 'Hair loss(Y/N)',
        'Pimples(Y/N)', 'Fast food (Y/N)',
        'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)',
        'BP _Diastolic (mmHg)', 'Follicle No. (L)',
        'Follicle No. (R)', 'Avg. F size (L) (mm)',
        'Avg. F size (R) (mm)', 'Endometrium (mm)'
    ])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of PCOS\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of PCOS\n\nProbability: {probability:.2%}")
