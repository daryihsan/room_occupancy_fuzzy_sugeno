# RoomSense

## Struktur folder

```text
Room_Occupancy_Fuzzy_Sugeno/
├── app_room_occupancy_fuzzy.py
├── requirements.txt
└── outputs/
    └── room_occupancy_fuzzy_sugeno_bundle.joblib
```

## Menjalankan aplikasi
```bash
pip install -r requirements.txt
streamlit run app_room_occupancy_fuzzy.py
```

## Tentang temperatur dan suara
Model hasil eksperimen terbaru memilih feature set `compact`:
- `light_mean_log`
- `co2_log`
- `pir_active_count`

Karena temperatur, suara, dan CO2 slope tidak termasuk fitur terpilih, perubahan sensor tersebut tidak mengubah prediksi. GUI menempatkannya pada bagian sensor tambahan dan memberi penjelasan eksplisit.

## Input banyak observasi
- Gunakan halaman **Prediksi banyak data**.
- Unduh template CSV, isi satu observasi per baris, lalu upload. (Data penelitian juga tersedia di `Occupancy_Estimation.csv`)
- Atau masukkan banyak baris melalui tabel input.
- Untuk model compact, kolom sensor wajib adalah S1–S4 Light, S5_CO2, S6_PIR, dan S7_PIR. Kolom sensor lain boleh ikut disertakan sebagai informasi tambahan.
