import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Judul
st.title("Aplikasi Prediksi Gaji Keamanan Siber")
st.sidebar.title("Navigasi")
st.sidebar.subheader("Menu")

# Load Model
filename = 'CybSec_Salaries.sav'
try:
    model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file 'CybSec_Salaries.sav' ada di direktori yang sama.")

# Pilih Menu
menu = st.sidebar.radio("Pilih menu:", options=["Deskripsi Dataset", "Visualisasi", "Prediksi Gaji"])

# Fungsi untuk memuat dataset
@st.cache
def load_data():
    try:
        data = pd.read_csv("salaries_cyber.csv")
        # Map pengalaman ke nilai numerik
        experience_mapping = {'EN': 1, 'MI': 3, 'SE': 7, 'EX': 10}
        data['years_experience'] = data['experience_level'].map(experience_mapping)
        return data
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return None

# Analisis Dataset
if menu == "Deskripsi Dataset":
    data = load_data()
    if data is not None:
        st.write("### Dataset Cybersecurity Salaries")
        st.dataframe(data.head())
        st.write("### Statistik Deskriptif")
        st.write(data.describe())
        st.write("### Informasi Dataset")
        st.text(data.info())

# Visualisasi Dataset
elif menu == "Visualisasi":
    data = load_data()
    if data is not None:
        # Distribusi Gaji
        st.write("### Distribusi Gaji dalam USD")
        fig, ax = plt.subplots()
        sns.histplot(data['salary_in_usd'], kde=True, ax=ax)
        ax.set_title("Distribusi Gaji dalam USD")
        st.pyplot(fig)

        # # Korelasi Antar Variabel
        # st.write("### Korelasi Antar Variabel")
        # fig, ax = plt.subplots(figsize=(8, 6))
        # sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        # ax.set_title("Korelasi Antar Variabel")
        # st.pyplot(fig)

        # Visualisasi dengan Plotly (Scatter)
        st.write("### Visualisasi Pengalaman vs Gaji")
        fig = px.scatter(data, x="years_experience", y="salary_in_usd",
                         title="Pengaruh Pengalaman terhadap Gaji dalam USD", 
                         labels={"years_experience": "Tahun Pengalaman", "salary_in_usd": "Gaji (USD)"})
        st.plotly_chart(fig)

        # Bar Chart: Rata-rata Gaji per Pekerjaan
        st.write("### Rata-rata Gaji per Pekerjaan")
        avg_salary_job = data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_salary_job.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Rata-rata Gaji per Pekerjaan")
        ax.set_xlabel("Pekerjaan")
        ax.set_ylabel("Gaji (USD)")
        st.pyplot(fig)

        # Treemap: Distribusi Gaji Berdasarkan Pekerjaan
        st.write("### Treemap Distribusi Gaji Berdasarkan Pekerjaan")
        fig = px.treemap(data, path=["job_title"], values="salary_in_usd",
                         title="Treemap: Distribusi Gaji Berdasarkan Pekerjaan",
                         labels={"job_title": "Pekerjaan", "salary_in_usd": "Gaji (USD)"})
        st.plotly_chart(fig)

# Prediksi Gaji
elif menu == "Prediksi Gaji":
    st.write("### Prediksi Gaji Berdasarkan Pengalaman")
    years_experience = st.number_input("Masukkan jumlah tahun pengalaman (contoh: 5)", min_value=0.0, step=0.1)
    if st.button("Prediksi"):
        try:
            prediction = model.predict([[years_experience]])
            st.success(f"Prediksi gaji Anda adalah: USD {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

# Sidebar Info
st.sidebar.info("Aplikasi ini digunakan untuk memprediksi gaji berdasarkan data pengalaman kerja di bidang keamanan siber.")
