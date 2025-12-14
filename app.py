import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(page_title="Prediksi Harga Cabai Merah Besar", layout="wide")

st.title("📈 Prediksi Harga Cabai Merah Besar Kabupaten Bekasi")
st.markdown("Menggunakan metode **SARIMA** dan **XGBoost**")

# =====================
# Load Model
# =====================
@st.cache_resource
def load_models():
    sarima_model = joblib.load("model/sarima.joblib")
    xgb_model = joblib.load("model/xgb.joblib")
    return sarima_model, xgb_model

sarima_model, xgb_model = load_models()

# =====================
# Load Dataset
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("data/cabai_bekasi.csv")
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    return df

df = load_data()

st.subheader("📊 Data Historis Harga Cabai")
st.dataframe(df.tail())

# =====================
# Pilih Model
# =====================
model_option = st.selectbox("Pilih Metode Prediksi", ("SARIMA", "XGBoost"))

# =====================
# Parameter Prediksi
# =====================
steps = st.number_input("Jumlah periode prediksi", min_value=1, max_value=30, value=7)

# =====================
# Proses Prediksi
# =====================
if st.button("🔮 Prediksi"):
    if model_option == "SARIMA":
        forecast = sarima_model.forecast(steps=steps)
        pred_df = pd.DataFrame({"Prediksi": forecast})

    else:
        # Contoh sederhana fitur lag
        last_values = df['harga'].values[-steps:]
        X_pred = np.array(last_values).reshape(steps, 1)
        forecast = xgb_model.predict(X_pred)
        pred_df = pd.DataFrame({"Prediksi": forecast})

    st.subheader("📈 Hasil Prediksi")

    fig, ax = plt.subplots()
    ax.plot(df['tanggal'], df['harga'], label="Data Aktual")
    ax.plot(pd.date_range(df['tanggal'].iloc[-1], periods=steps+1, freq='D')[1:], forecast, label="Prediksi")
    ax.legend()

    st.pyplot(fig)
    st.dataframe(pred_df)

    st.download_button(
        label="⬇️ Download Hasil Prediksi",
        data=pred_df.to_csv(index=False),
        file_name="hasil_prediksi_cabai.csv",
        mime="text/csv"
    )
