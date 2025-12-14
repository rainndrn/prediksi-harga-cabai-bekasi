# 📈 Prediksi Harga Cabai Merah Besar Kabupaten Bekasi

Aplikasi web ini dibuat untuk melakukan **prediksi harga Cabai Merah Besar di Kabupaten Bekasi** menggunakan dua metode peramalan, yaitu **SARIMA** dan **XGBoost**. Aplikasi dibangun dengan **Streamlit** dan dideploy menggunakan **Streamlit Community Cloud**.

Penelitian ini bertujuan untuk membantu memberikan gambaran harga cabai di masa mendatang berdasarkan data historis, sehingga dapat menjadi bahan pertimbangan bagi masyarakat maupun pihak terkait.

---

## 🚀 Fitur Aplikasi

* Menampilkan data historis harga cabai merah besar
* Prediksi harga menggunakan metode:

  * SARIMA (Statistical Time Series)
  * XGBoost (Machine Learning)
* Visualisasi grafik data aktual dan hasil prediksi
* Tabel hasil prediksi
* Download hasil prediksi dalam format CSV

---

## 🗂️ Struktur Folder Project

```
prediksi-harga-cabai-bekasi/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── sarima.joblib
│   └── xgb.joblib
│
└── data/
    └── cabai_bekasi.csv
```

---

## 📊 Dataset

Dataset disimpan dalam folder `data/` dengan format CSV dan memiliki kolom utama:

* `tanggal` : tanggal pencatatan harga
* `harga`   : harga cabai merah besar

Contoh format:

| tanggal    | harga |
| ---------- | ----- |
| 2023-01-01 | 55000 |
| 2023-01-02 | 56000 |

---

## ⚙️ Library yang Digunakan

* streamlit
* pandas
* numpy
* matplotlib
* joblib
* statsmodels
* xgboost
* scikit-learn

Semua dependency tercantum dalam file `requirements.txt`.

---

## ▶️ Cara Menjalankan Secara Lokal

1. Clone repository

```bash
git clone https://github.com/username/prediksi-harga-cabai-bekasi.git
cd prediksi-harga-cabai-bekasi
```

2. Install dependency

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi

```bash
streamlit run app.py
```

---

## ☁️ Deploy ke Streamlit Cloud

1. Upload project ke GitHub (Public Repository)
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Login menggunakan akun GitHub
4. Klik **New App**
5. Pilih repository dan file utama `app.py`
6. Klik **Deploy** dan tunggu proses selesai

---

## 📌 Catatan

* Model SARIMA dan XGBoost sudah dilatih sebelumnya dan disimpan dalam format `.joblib`
* Aplikasi ini ditujukan untuk keperluan **penelitian dan akademik**

---

## 👨‍🎓 Penulis

**Nama:** *Raina Andriani Putri*
**Judul Penelitian:**
**Prediksi Harga Cabai Merah Besar di Kabupaten Bekasi Menggunakan Metode SARIMA dan XGBoost**

---

Jika ada pengembangan lanjutan atau penyesuaian model, silakan modifikasi file `app.py` sesuai kebutuhan.
