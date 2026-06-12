# ChickPoo

ChickPoo adalah demo screening awal kesehatan ayam berbasis foto feses. Frontend dibuat dengan React dan backend prediksi dibuat dengan FastAPI.

## Jalankan Lokal

Backend:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Model yang dipakai berada di `models/best_model.keras`. Frontend membaca alamat backend dari `VITE_API_URL`, sehingga nanti bisa disiapkan untuk Vercel dengan backend terpisah.
# ChickPoo

ChickPoo adalah aplikasi screening awal penyakit ayam dari foto feses. Project ini memakai model machine learning untuk memprediksi empat kondisi: Sehat, Coccidiosis, Newcastle Disease, dan Salmonella.

## Stack

- Backend: FastAPI
- Frontend: React + Vite
- Model: Keras

## Struktur Utama

- `backend/`: API prediksi dan service model.
- `frontend/`: aplikasi web ChickPoo.
- `models/best_model.keras`: model terbaik yang dipakai aplikasi.
- `models/feces_reference_stats.npz`: referensi embedding untuk validasi input non-feses.
- `notebook_training.ipynb`: notebook training model.
- `scripts/`: script pendukung eksperimen dan preprocessing.

## Menjalankan Backend

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
http://127.0.0.1:8000/health
```

## Menjalankan Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend berjalan di:

```bash
http://127.0.0.1:5173/
```

## Catatan Dataset

Dataset mentah dan file zip berukuran besar tidak dimasukkan ke repository GitHub. File yang disimpan di repository difokuskan untuk menjalankan aplikasi, membaca notebook training, dan melakukan pengembangan lanjutan.
