# ChickPoo

ChickPoo adalah aplikasi screening awal penyakit ayam dari foto feses. Aplikasi ini memakai React untuk frontend dan FastAPI untuk backend prediksi.

## Alur Prediksi

ChickPoo sekarang memakai pipeline model bertingkat:

1. Validasi objek: `models/object_validation_model.keras` mengecek apakah foto cukup meyakinkan sebagai feses ayam.
2. Klasifikasi penyakit: `models/best_model.keras` hanya dijalankan jika foto lolos validasi objek.
3. Hasil akhir menampilkan prediksi penyakit, persentase tiap kelas, status validasi objek, dan rekomendasi tindakan.

Jika foto tidak lolos validasi objek, backend tidak memaksa model penyakit untuk menebak. Aplikasi akan meminta pengguna mengambil ulang foto yang lebih sesuai.

## Stack

- Backend: FastAPI
- Frontend: React + Vite
- Model: Keras / TensorFlow

## Struktur Utama

- `backend/`: API prediksi dan service model.
- `frontend/`: aplikasi web ChickPoo.
- `models/best_model.keras`: model klasifikasi penyakit ayam.
- `models/object_validation_model.keras`: model validasi feses ayam atau bukan.
- `models/object_validation_metadata.json`: metadata threshold dan performa model validasi objek.
- `notebook_training.ipynb`: notebook training model penyakit.
- `code-object-validation.ipynb`: notebook training validasi objek.
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
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend berjalan di:

```bash
http://127.0.0.1:5173/
```

Frontend membaca alamat backend dari `VITE_API_URL`, sehingga nanti bisa disiapkan untuk Vercel dengan backend terpisah.

## Deployment

Rekomendasi deployment saat ini:

1. Deploy backend FastAPI ke layanan server Python seperti Render atau Railway.
2. Deploy frontend React ke Vercel.
3. Isi environment variable frontend `VITE_API_URL` dengan URL backend production.
4. Isi environment variable backend `CORS_ORIGINS` dengan URL frontend production, misalnya `https://nama-project.vercel.app`.

Backend sebaiknya tidak diletakkan di Vercel Serverless karena aplikasi memuat TensorFlow dan model Keras. Agar perilaku mendekati lokal, gunakan layanan backend yang bisa menjalankan proses Python persistent dan memiliki memori cukup untuk memuat dua model.

Contoh command backend production:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Contoh build frontend:

```bash
cd frontend
npm install
npm run build
```

## Catatan Dataset

Dataset mentah dan file zip berukuran besar tidak dimasukkan ke repository GitHub. File yang disimpan di repository difokuskan untuk menjalankan aplikasi, membaca notebook training, dan melakukan pengembangan lanjutan.
