import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# from plotly.offline import init_notebook_mode
import seaborn as sns
import datetime as dt
import warnings
# import plotly.graph_objects as go
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pickle
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
# init_notebook_mode(connected=True)

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
