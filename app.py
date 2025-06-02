import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Load model, scaler, and column names
model = joblib.load('best_rf.joblib')

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    column_names = pickle.load(f)

# Streamlit UI
st.title("Prediksi Kemungkinan Resign Karyawan")

# Buat dua kolom
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
    travel_freq = st.selectbox("Frekuensi Perjalanan Dinas", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Departemen", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Peran Pekerjaan", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", 
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])
    education_field = st.selectbox("Bidang Pendidikan", [
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
    ])
    overtime = st.selectbox("Lembur", ["Yes", "No"])

with col2:
    performance_rating = st.selectbox("Penilaian Kinerja", [1, 2, 3, 4])
    education = st.slider("Tingkat Pendidikan", 1, 5, 3)
    environment_satisfaction = st.slider("Kepuasan Lingkungan Kerja", 1, 4, 3)
    job_involvement = st.slider("Keterlibatan dalam Pekerjaan", 1, 4, 2)
    job_level = st.slider("Level Jabatan", 1, 5, 1)
    job_satisfaction = st.slider("Kepuasan Kerja", 1, 4, 3)
    relationship_satisfaction = st.slider("Kepuasan Relasi", 1, 4, 2)
    stock_option_level = st.slider("Level Opsi Saham", 0, 3, 0)
    work_life_balance = st.slider("Keseimbangan Kerja-Hidup", 1, 4, 3)

# Step 1: Build a one-row DataFrame
user_input = {
    "Gender": 1 if gender == "Male" else 0,
    "OverTime": 1 if overtime == "Yes" else 0,
    "PerformanceRating": performance_rating,
    "BusinessTravel": travel_freq,
    "Department": department,
    "EducationField": education_field,
    "JobRole": job_role,
    "MaritalStatus": marital_status,
    "Education": education,
    "EnvironmentSatisfaction": environment_satisfaction,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobSatisfaction": job_satisfaction,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StockOptionLevel": stock_option_level,
    "WorkLifeBalance": work_life_balance
}

df = pd.DataFrame([user_input])

# Step 2: One-hot encode categorical variables
nominals = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=nominals)

# Step 3: Align with training columns
df = df.reindex(columns=column_names, fill_value=0)

# Step 4: Scale
df_scaled = scaler.transform(df)

# Step 5: Predict
if st.button("Prediksi Attrition"):
    prediction = model.predict(df_scaled)[0]
    if prediction > 0.5:
        st.error(f"Karyawan kemungkinan **resign** (skor: {prediction:.2f})")
    else:
        st.success(f"Karyawan kemungkinan **bertahan** (skor: {prediction:.2f})")