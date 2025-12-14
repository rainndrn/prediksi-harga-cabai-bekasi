import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import os

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
    try:
        sarima_model = joblib.load("model/sarima.joblib")
        xgb_model = joblib.load("model/xgb.joblib")
        return sarima_model, xgb_model
    except FileNotFoundError as e:
        st.error(f"Model file tidak ditemukan: {e}")
        st.info("Pastikan folder 'model' berisi file 'sarima.joblib' dan 'xgb.joblib'")
        st.stop()

sarima_model, xgb_model = load_models()

# =====================
# Load Dataset
# =====================
@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    path = "data/cabai-merah-besar.csv"

    if not os.path.exists(path):
        st.error("File dataset tidak ditemukan!")
        st.info(f"Pastikan file ada di: {path}")
        st.info(f"Current directory: {os.getcwd()}")
        st.stop()

    # Dataset kamu pakai delimiter ;
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # rapikan nama kolom
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.lower()
        .str.strip()
    )

    # validasi kolom wajib
    if not {"tanggal", "harga"}.issubset(df.columns):
        st.error("Kolom wajib 'tanggal' dan 'harga' tidak ditemukan!")
        st.info(f"Kolom terbaca: {df.columns.tolist()}")
        st.stop()

    # parsing tanggal & harga
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    # buang data invalid
    df = (
        df.dropna(subset=["tanggal", "harga"])
          .sort_values("tanggal")
          .reset_index(drop=True)
    )

    return df
    
    # Jika tidak ada file yang ditemukan
    st.error("File 'cabai-merah-besar.csv' tidak ditemukan!")
    st.info(f"Path yang dicoba: {possible_paths}")
    st.info(f"Current directory: {os.getcwd()}")
    st.info(f"Files in current directory: {os.listdir('.')}")
    if os.path.exists('data'):
        st.info(f"Files in data directory: {os.listdir('data')}")
    st.stop()

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
    try:
        if model_option == "SARIMA":
            forecast = sarima_model.forecast(steps=steps)
            pred_df = pd.DataFrame({"Prediksi": forecast})
        else:
            # ===== XGBoost (FIX feature mismatch) =====
            n_feat = getattr(xgb_model, "n_features_in_", 3)
        
            history = df["harga"].dropna().values.tolist()
        
            if len(history) < n_feat:
                st.error(f"Data tidak cukup untuk XGBoost. Butuh minimal {n_feat} data terakhir.")
                st.stop()
        
            preds = []
            for _ in range(steps):
                # fitur = lag1, lag2, ..., lag_n
                X_input = np.array(history[-n_feat:][::-1], dtype=float).reshape(1, n_feat)
                y_hat = float(xgb_model.predict(X_input)[0])
                preds.append(y_hat)
                history.append(y_hat)
        
            forecast = np.array(preds)
            pred_df = pd.DataFrame({"Prediksi": forecast})
        
        st.subheader("📈 Hasil Prediksi")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['tanggal'], df['harga'], label="Data Aktual", linewidth=2)
        
        # Generate tanggal untuk prediksi
        last_date = df['tanggal'].iloc[-1]
        pred_dates = pd.date_range(start=last_date, periods=steps+1, freq='D')[1:]
        
        ax.plot(pred_dates, forecast, label="Prediksi", linewidth=2, linestyle='--', color='red')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (Rp)")
        ax.set_title(f"Prediksi Harga Cabai dengan {model_option}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Tambahkan tanggal ke dataframe prediksi
        pred_df['Tanggal'] = pred_dates
        pred_df = pred_df[['Tanggal', 'Prediksi']]
        
        st.dataframe(pred_df)
        
        st.download_button(
            label="⬇️ Download Hasil Prediksi",
            data=pred_df.to_csv(index=False),
            file_name="hasil_prediksi_cabai.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.info("Pastikan model telah dilatih dengan benar dan data input sesuai format.")
