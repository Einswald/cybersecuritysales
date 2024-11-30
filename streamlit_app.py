import streamlit as st
import pickle

# Load model dari file .sav
filename = 'CybSec_Salaries.sav'
try:
    model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file 'CybSec_Salaries.sav' ada di direktori yang sama.")

# Judul Aplikasi
st.title("Aplikasi Prediksi Gaji Keamanan Siber")
st.write("Masukkan data untuk melihat prediksi gaji.")

# Input dari pengguna
years_experience = st.number_input(
    "Masukkan jumlah tahun pengalaman (contoh: 5)",
    min_value=0.0,
    step=0.1
)

# Prediksi
if st.button("Prediksi"):
    try:
        prediction = model.predict([[years_experience]])
        st.success(f"Prediksi gaji Anda adalah: Rp {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
