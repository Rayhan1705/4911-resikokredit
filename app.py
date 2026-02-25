import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💰",
    layout="wide"
)

# --- LOAD MODEL & PREPROCESSING TOOLS ---
@st.cache_resource
def load_assets():
    try:
        model_xgb = joblib.load('model_xgboost.pkl')
        model_rf = joblib.load('model_random_forest.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model_xgb, model_rf, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return None, None, None

model_xgb, model_rf, scaler, encoders = load_assets()

# --- JUDUL & DESKRIPSI ---
St.markdown("Hossain Wismaya Rayhan 22.11.4911")
st.title("💰 Sistem Prediksi Risiko Kredit Nasabah")
st.markdown("""
Aplikasi ini menggunakan Machine Learning (**XGBoost & Random Forest**) untuk memprediksi apakah nasabah 
berpotensi **Gagal Bayar (1)** atau **Lancar (0)**.
""")

st.sidebar.header("🔧 Panel Input Data")

# --- INPUT USER (SIDEBAR) ---
def user_input_features():
    st.sidebar.markdown("### 1. Data Demografi & Pekerjaan")
    
    # Input Usia
    person_age = st.sidebar.number_input("Usia Nasabah (Tahun)", min_value=18, max_value=100, value=25)
    
    # Input Pendapatan
    person_income = st.sidebar.number_input("Pendapatan Tahunan ($)", min_value=1000, max_value=10000000, value=50000, step=1000)
    
    # Input Lama Kerja
    person_emp_length = st.sidebar.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=60, value=2, step=1)

    # Asumsi: Seseorang minimal mulai bekerja umur 15 tahun (part-time dsb).
    # Jadi Lama Kerja tidak boleh lebih besar dari (Usia - 15).
    is_valid_input = True
    if person_emp_length > (person_age - 15):
        st.sidebar.error(f"⚠️ Data Tidak Logis: Usia {person_age} tahun tidak mungkin memiliki pengalaman kerja {person_emp_length} tahun.")
        is_valid_input = False
    # ---------------------------------

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. Data Pinjaman")

    loan_amnt = st.sidebar.number_input("Jumlah Pinjaman ($)", min_value=500, max_value=500000, value=10000, step=500)

    # Data Kategorikal
    home_options = {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}
    person_home_ownership = st.sidebar.selectbox("Status Kepemilikan Rumah", list(home_options.keys()))
    
    intent_options = {
        'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 
        'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5
    }
    loan_intent = st.sidebar.selectbox("Tujuan Pinjaman", list(intent_options.keys()))
    
    grade_options = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    loan_grade = st.sidebar.selectbox("Nilai Pinjaman (Grade)", list(grade_options.keys()))
    

    # Kalkulasi Otomatis
    if person_income > 0:
        loan_percent_income = loan_amnt / person_income
    else:
        loan_percent_income = 0.0


    # Return data DAN status validasi
    data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'person_home_ownership': home_options[person_home_ownership],
        'loan_amnt': loan_amnt,
        'loan_grade': grade_options[loan_grade],
        'loan_intent': intent_options[loan_intent],
        'loan_percent_income': loan_percent_income
    }
    
    return data, is_valid_input

# Memanggil fungsi input dan menangkap status validasi
input_data, is_valid = user_input_features()

# --- TAMPILAN INPUT DI HALAMAN UTAMA ---
st.subheader("📋 Ringkasan Data Nasabah")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Usia", f"{input_data['person_age']} Tahun")
    st.metric("Pendapatan", f"${input_data['person_income']:,}")
    # Warna merah jika data tidak valid
    if not is_valid:
        st.markdown(f":red[**Lama Kerja: {input_data['person_emp_length']} Tahun (Tidak Valid)**]")
    else:
        st.metric("Lama Kerja", f"{input_data['person_emp_length']} Tahun")

with col2:
    st.metric("Jumlah Pinjaman", f"${input_data['loan_amnt']:,}")
    home_text = [k for k, v in {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}.items() if v == input_data['person_home_ownership']][0]
    st.metric(f"Rumah: ", home_text)

with col3:
    grade_text = [k for k, v in {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}.items() if v == input_data['loan_grade']][0]
    st.metric("Loan Grade", grade_text)
    intent_map = {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}
    intent_text = [k for k, v in intent_map.items() if v == input_data['loan_intent']][0]

st.markdown("---")

# --- PREDICTION SECTION ---
st.subheader("🔍 Hasil Prediksi Risiko")

model_choice = st.selectbox("Pilih Model untuk Prediksi:", ["XGBoost (Recommended)", "RANDOM FOREST"])

# LOGIC PENGAMAN TOMBOL
if is_valid:
    # Tombol hanya aktif jika data valid
    if st.button("JALANKAN PREDIKSI"):
        if model_xgb is not None and scaler is not None:
            
            # Urutan Fitur Sesuai Training
            final_features = [
                input_data['person_age'],
                input_data['person_income'],
                input_data['person_emp_length'],
                input_data['person_home_ownership'],
                input_data['loan_amnt'],
                input_data['loan_grade'],
                input_data['loan_intent'],
                input_data['loan_percent_income']
            ]
            
            # Scaling & Prediksi
            features_array = np.array([final_features])
            features_scaled = scaler.transform(features_array)
            
            if model_choice == "XGBoost (Recommended)":
                prediction = model_xgb.predict(features_scaled)
                probability = model_xgb.predict_proba(features_scaled)
            else:
                prediction = model_rf.predict(features_scaled)
                probability = model_rf.predict_proba(features_scaled)
                
            # Tampilkan Hasil
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                if prediction[0] == 0:
                    st.success("✅ **Status: LANCAR**")
                    st.success("**Nasabah diprediksi mampu membayar pinjaman.**")
                else:
                    st.error("⚠️ **Status: BERISIKO GAGAL BAYAR**")
                    st.error("**Nasabah memiliki risiko untuk gagal bayar pinjaman.**")
            
            with col_res2:
                prob_default = probability[0][1]
                st.write(f"📊**Probabilitas Gagal Bayar:**")
                if prob_default > 0.5:
                    st.progress(float(prob_default), text="Risiko Tinggi")
                else:
                    st.progress(float(prob_default), text="Risiko Rendah")
                st.write(f"{prob_default:.2%}")
        else:
            st.error("Model belum dimuat. Cek file .pkl Anda.")
else:
    # Jika data tidak valid, tombol hilang dan muncul peringatan

    st.warning("⚠️ **PERHATIAN:** Mohon perbaiki data input di Sidebar. Lama bekerja tidak boleh melebihi usia nasabah (dikurangi usia wajar mulai bekerja).")

