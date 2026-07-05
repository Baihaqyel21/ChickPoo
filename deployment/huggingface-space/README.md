---
title: ChickPoo API
emoji: 🐔
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ChickPoo API

FastAPI backend untuk validasi foto feses ayam dan screening awal kondisi kesehatan ayam.

Endpoint utama:

- `GET /health`
- `POST /warmup`
- `POST /predict`

Frontend React/Vercel harus mengarah ke URL Space ini melalui `VITE_API_URL`.
