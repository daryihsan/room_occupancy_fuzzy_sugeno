from pathlib import Path
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. Konfigurasi aplikasi dan tampilan
# ============================================================
st.set_page_config(
    page_title="RoomSense | Fuzzy Occupancy",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14243a 0%, #1c3553 100%);
    }
    [data-testid="stSidebar"] * {
        color: #eef4fb;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d8e3ef;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {
        color: #ffffff !important;
        background-color: #294b70 !important;
        border: 1px solid #527ba5 !important;
        border-radius: 6px !important;
        padding: 0.15rem 0.4rem !important;
        font-weight: 650 !important;
    }
    .hero {
        padding: 1.35rem 1.55rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #17324f 0%, #245b78 58%, #27828a 100%);
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 26px rgba(24, 55, 82, .14);
    }
    .hero h1 {
        color: white !important;
        margin: 0 0 .35rem 0;
        font-size: 2rem;
        letter-spacing: -.03em;
    }
    .hero p {
        color: #e0edf5 !important;
        margin: 0;
        font-size: .98rem;
    }
    .section-card {
        background: white;
        border: 1px solid #e5ebf2;
        border-radius: 15px;
        padding: 1rem 1.1rem;
        margin-bottom: .75rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5ebf2;
        border-radius: 14px;
        padding: .85rem 1rem;
        box-shadow: 0 2px 9px rgba(20, 45, 70, .035);
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5ebf2;
        border-radius: 15px;
        padding: 1rem 1.1rem;
    }
    .small-muted {
        color: #64748b;
        font-size: .86rem;
    }
    .pill {
        display: inline-block;
        padding: .22rem .58rem;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 650;
        margin: .12rem .18rem .12rem 0;
    }
    .pill-used {
        background: #dcfce7;
        color: #166534;
    }
    .pill-unused {
        background: #e9eef5;
        color: #526174;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.65rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e5ebf2;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. Lokasi dan pemuatan bundle
# ============================================================
APP_DIR = Path(__file__).resolve().parent
BUNDLE_RELATIVE = Path("outputs/room_occupancy_fuzzy_sugeno_bundle.joblib")

def locate_bundle():
    candidates = [
        APP_DIR / BUNDLE_RELATIVE,
        APP_DIR / "room_occupancy_fuzzy_sugeno_bundle.joblib",
        Path.cwd() / BUNDLE_RELATIVE,
        Path.cwd() / "room_occupancy_fuzzy_sugeno_bundle.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

@st.cache_resource(show_spinner="Memuat model fuzzy...")
def load_bundle(path_string):
    return joblib.load(path_string)

bundle_path = locate_bundle()
if bundle_path is None:
    st.error(
        "Bundle model belum ditemukan. Letakkan `room_occupancy_fuzzy_sugeno_bundle.joblib` "
        "di dalam folder `outputs/` yang sejajar dengan file aplikasi."
    )
    st.info(
        "Jalankan notebook sampai bagian ekspor model, bundle portable akan menyimpan parameter preprocessing sebagai array numerik."
    )
    st.stop()

try:
    bundle = load_bundle(str(bundle_path))
except Exception as exc:
    st.error("Bundle model gagal dibaca.")
    st.exception(exc)
    st.warning(
        "Jika error terjadi saat joblib.load(), kemungkinan bundle dibuat dengan versi library "
        "yang tidak kompatibel. Buat ulang bundle."
    )
    st.stop()

# ============================================================
# 3. Informasi model dan sensor
# ============================================================
feature_names = list(bundle["feature_names"])
raw_defaults = dict(bundle.get("raw_sensor_defaults", {}))

SENSOR_COLUMNS = (
    [f"S{i}_Temp" for i in range(1, 5)]
    + [f"S{i}_Light" for i in range(1, 5)]
    + [f"S{i}_Sound" for i in range(1, 5)]
    + ["S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
)

DEFAULT_FALLBACKS = {
    "S1_Temp": 24.0, "S2_Temp": 24.0, "S3_Temp": 24.0, "S4_Temp": 24.0,
    "S1_Light": 100.0, "S2_Light": 100.0, "S3_Light": 100.0, "S4_Light": 100.0,
    "S1_Sound": 0.1, "S2_Sound": 0.1, "S3_Sound": 0.1, "S4_Sound": 0.1,
    "S5_CO2": 450.0, "S5_CO2_Slope": 0.0, "S6_PIR": 0, "S7_PIR": 0,
}
for key, value in DEFAULT_FALLBACKS.items():
    raw_defaults.setdefault(key, value)

def feature_groups(names):
    names = set(names)
    return {
        "temperature": any(n.startswith("temp_") for n in names),
        "light": any(n.startswith("light_") for n in names),
        "sound": any(n.startswith("sound_") for n in names),
        "co2": bool(names.intersection({"co2_ppm", "co2_log"})),
        "co2_slope": "co2_slope" in names,
        "pir": "pir_active_count" in names,
    }

groups = feature_groups(feature_names)

def required_raw_columns(names):
    g = feature_groups(names)
    cols = []
    if g["temperature"]:
        cols += [f"S{i}_Temp" for i in range(1, 5)]
    if g["light"]:
        cols += [f"S{i}_Light" for i in range(1, 5)]
    if g["sound"]:
        cols += [f"S{i}_Sound" for i in range(1, 5)]
    if g["co2"]:
        cols += ["S5_CO2"]
    if g["co2_slope"]:
        cols += ["S5_CO2_Slope"]
    if g["pir"]:
        cols += ["S6_PIR", "S7_PIR"]
    return cols

required_columns = required_raw_columns(feature_names)

def build_features(raw):
    temp = np.asarray([raw[f"S{i}_Temp"] for i in range(1, 5)], dtype=float)
    light = np.asarray([raw[f"S{i}_Light"] for i in range(1, 5)], dtype=float)
    sound = np.asarray([raw[f"S{i}_Sound"] for i in range(1, 5)], dtype=float)

    def safe_mean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    def safe_std(x):
        return float(np.nanstd(x, ddof=0)) if np.isfinite(x).any() else np.nan

    temp_mean, temp_std = safe_mean(temp), safe_std(temp)
    light_mean, light_std = safe_mean(light), safe_std(light)
    sound_mean, sound_std = safe_mean(sound), safe_std(sound)
    co2 = float(raw["S5_CO2"])
    co2_slope = float(raw["S5_CO2_Slope"])
    pir_count = int(raw["S6_PIR"]) + int(raw["S7_PIR"])

    return {
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "light_mean": light_mean,
        "light_std": light_std,
        "sound_mean": sound_mean,
        "sound_std": sound_std,
        "co2_ppm": co2,
        "co2_slope": co2_slope,
        "pir_active_count": pir_count,
        "light_mean_log": np.log1p(max(light_mean, 0.0)) if np.isfinite(light_mean) else np.nan,
        "light_std_log": np.log1p(max(light_std, 0.0)) if np.isfinite(light_std) else np.nan,
        "sound_mean_log": np.log1p(max(sound_mean, 0.0)) if np.isfinite(sound_mean) else np.nan,
        "sound_std_log": np.log1p(max(sound_std, 0.0)) if np.isfinite(sound_std) else np.nan,
        "co2_log": np.log1p(max(co2, 0.0)) if np.isfinite(co2) else np.nan,
    }

# ============================================================
# 4. Preprocessing kompatibel
# ============================================================
def get_preprocessing_parameters(model_bundle):
    """
    Bundle portable memakai array numerik.
    """
    if "imputer_statistics" in model_bundle:
        imputer_statistics = np.asarray(model_bundle["imputer_statistics"], dtype=float)
    else:
        imputer_statistics = np.asarray(
            model_bundle["imputer"].statistics_, dtype=float
        )

    if "x_scaler_center" in model_bundle:
        x_center = np.asarray(model_bundle["x_scaler_center"], dtype=float)
        x_scale = np.asarray(model_bundle["x_scaler_scale"], dtype=float)
    else:
        x_scaler = model_bundle["x_scaler"]
        center_attr = getattr(x_scaler, "center_", None)
        scale_attr = getattr(x_scaler, "scale_", None)
        x_center = (
            np.zeros(len(feature_names), dtype=float)
            if center_attr is None else np.asarray(center_attr, dtype=float)
        )
        x_scale = (
            np.ones(len(feature_names), dtype=float)
            if scale_attr is None else np.asarray(scale_attr, dtype=float)
        )

    if "y_scaler_mean" in model_bundle:
        y_mean = float(model_bundle["y_scaler_mean"])
        y_scale = float(model_bundle["y_scaler_scale"])
    else:
        y_scaler = model_bundle["y_scaler"]
        y_mean = float(np.asarray(y_scaler.mean_).ravel()[0])
        y_scale = float(np.asarray(y_scaler.scale_).ravel()[0])

    x_scale = np.where(np.abs(x_scale) < 1e-12, 1.0, x_scale)
    y_scale = y_scale if abs(y_scale) >= 1e-12 else 1.0
    return imputer_statistics, x_center, x_scale, y_mean, y_scale

def regularize_covariance(cov, reg=1e-2):
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    scale = max(float(np.trace(cov) / cov.shape[0]), 1e-6)
    eigen_floor = max(reg * scale, 1e-6)
    eigenvalues = np.maximum(eigenvalues, eigen_floor)
    cov_reg = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (cov_reg + cov_reg.T) / 2.0

def antecedent_membership(X_scaled, centers_x, covariances_x):
    X_scaled = np.asarray(X_scaled, dtype=float)
    log_membership = np.empty((len(X_scaled), len(centers_x)), dtype=float)
    for i in range(len(centers_x)):
        cov = regularize_covariance(covariances_x[i], reg=1e-2)
        inv_cov = np.linalg.inv(cov)
        diff = X_scaled - centers_x[i]
        d2 = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
        log_membership[:, i] = -0.5 * np.maximum(d2, 0.0)
    log_membership -= log_membership.max(axis=1, keepdims=True)
    mu = np.exp(log_membership)
    mu /= np.maximum(mu.sum(axis=1, keepdims=True), 1e-12)
    return mu

def predict_one(model_bundle, raw):
    values = build_features(raw)
    row = np.asarray([[values[name] for name in feature_names]], dtype=float)

    stats, x_center, x_scale, y_mean, y_scale = get_preprocessing_parameters(model_bundle)

    # Median imputation performed explicitly using the stored training statistics.
    X_imputed = row.copy()
    missing = ~np.isfinite(X_imputed)
    if missing.any():
        col_indices = np.where(missing)[1]
        X_imputed[missing] = stats[col_indices]

    # RobustScaler transform performed explicitly from stored center_ and scale_.
    X_scaled = (X_imputed - x_center.reshape(1, -1)) / x_scale.reshape(1, -1)

    firing = antecedent_membership(
        X_scaled, model_bundle["centers_x"], model_bundle["covariances_x"]
    )
    phi = np.column_stack([X_scaled, np.ones(len(X_scaled))])
    local_scaled = phi @ np.asarray(model_bundle["consequents_scaled"], dtype=float).T
    prediction_scaled = np.sum(firing * local_scaled, axis=1)

    prediction_raw = float(prediction_scaled[0] * y_scale + y_mean)
    prediction_bounded = float(np.clip(prediction_raw, 0.0, 3.0))
    prediction_count = int(np.floor(prediction_bounded + 0.5))
    local_outputs = local_scaled[0] * y_scale + y_mean

    return {
        "features": values,
        "firing": firing[0],
        "local_outputs": local_outputs,
        "contributions": firing[0] * local_outputs,
        "prediction_raw": prediction_raw,
        "prediction_bounded": prediction_bounded,
        "prediction_count": prediction_count,
    }

# ============================================================
# 5. Komponen antarmuka
# ============================================================
def sensor_input_group(title, columns, defaults, step, fmt, active=True):
    st.markdown(f"**{title}**")
    cols = st.columns(len(columns))
    values = {}
    for col, name in zip(cols, columns):
        with col:
            values[name] = st.number_input(
                name.replace("_", " "),
                value=float(defaults[name]),
                step=step,
                format=fmt,
                key=f"input_{name}",
            )
    return values

def make_raw_defaults():
    return {key: float(value) for key, value in raw_defaults.items()}

def raw_template_dataframe(n_rows=1):
    row = make_raw_defaults()
    return pd.DataFrame([{key: row[key] for key in SENSOR_COLUMNS} for _ in range(n_rows)])

def make_csv_template():
    row = make_raw_defaults()
    data = {"Date": ["2026-01-01"], "Time": ["08:00:00"]}
    data.update({key: [row[key]] for key in SENSOR_COLUMNS})
    return pd.DataFrame(data)

def run_batch_dataframe(input_df):
    rows = []
    rule_count = int(bundle["n_rules"])
    for row_idx, source_row in input_df.iterrows():
        raw = make_raw_defaults()
        for col in SENSOR_COLUMNS:
            if col in source_row.index and pd.notna(source_row[col]):
                raw[col] = float(source_row[col])
        # PIR is binary; reject invalid sensor states instead of silently rounding.
        for col in ["S6_PIR", "S7_PIR"]:
            if raw[col] not in (0.0, 1.0):
                raise ValueError(f"{col} harus bernilai 0 atau 1 (baris {row_idx + 1}).")

        result = predict_one(bundle, raw)
        output = source_row.to_dict()
        output["prediksi_raw"] = result["prediction_raw"]
        output["prediksi_bounded"] = result["prediction_bounded"]
        output["prediksi_jumlah_penghuni"] = result["prediction_count"]
        output["aturan_dominan"] = f"R{int(np.argmax(result['firing'])) + 1}"
        output["firing_maksimum"] = float(np.max(result["firing"]))
        for j in range(rule_count):
            output[f"firing_R{j+1}"] = float(result["firing"][j])
        rows.append(output)
    return pd.DataFrame(rows)

# ============================================================
# 6. Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🏠 RoomSense")
    st.caption("Fuzzy occupancy demonstrator")
    page = st.radio(
        "Navigasi",
        ["Prediksi tunggal", "Prediksi banyak data", "Tentang model"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Konfigurasi terpilih")
    st.write(f"**Metode**  `{str(bundle.get('selected_method', 'GK')).upper()}`")
    st.write(f"**Jumlah aturan**  `{bundle.get('n_rules', '?')}`")
    st.write(f"**Fuzzifier**  `{bundle.get('m', '?')}`")
    st.write(f"**Ridge alpha**  `{bundle.get('ridge_alpha', '?')}`")
    st.write(f"**Feature set**  `{bundle.get('selected_feature_set', 'N/A')}`")
    st.divider()
    st.caption("Model diidentifikasi dari dataset eksperimen satu ruangan. Hasil pada ruangan lain belum dijamin.")
    st.caption(f"Bundle: {bundle_path.name}")

# ============================================================
# 7. Header utama
# ============================================================
st.markdown(
    """
    <div class="hero">
      <h1>Room Occupancy Estimation</h1>
      <p>Estimasi jumlah penghuni menggunakan Fuzzy Clustering dan First-Order Sugeno.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

used_pills = " ".join(
    f'<span class="pill pill-used">{name}</span>' for name in feature_names
)
st.markdown(
    f'<div class="small-muted">Fitur yang digunakan model: {used_pills}</div>',
    unsafe_allow_html=True,
)

# ============================================================
# 8. Prediksi tunggal
# ============================================================
if page == "Prediksi tunggal":
    st.subheader("Input pembacaan sensor")
    st.write(
        "Masukkan nilai sensor mentah. Aplikasi menghitung fitur agregat dan transformasi "
        "secara otomatis mengikuti pipeline penelitian."
    )

    with st.form("single_prediction_form"):
        raw = make_raw_defaults()

        # Sensor yang memang berkontribusi terhadap feature set final.
        st.markdown("#### Sensor yang digunakan model")
        if groups["temperature"]:
            raw.update(sensor_input_group(
                "Temperatur S1-S4 (°C)", [f"S{i}_Temp" for i in range(1, 5)],
                raw_defaults, 0.1, "%.2f"
            ))
        if groups["light"]:
            raw.update(sensor_input_group(
                "Cahaya S1-S4 (lux)", [f"S{i}_Light" for i in range(1, 5)],
                raw_defaults, 1.0, "%.1f"
            ))
        if groups["sound"]:
            raw.update(sensor_input_group(
                "Suara S1-S4 (volt)", [f"S{i}_Sound" for i in range(1, 5)],
                raw_defaults, 0.01, "%.3f"
            ))
        if groups["co2"]:
            raw["S5_CO2"] = st.number_input(
                "CO₂ S5 (ppm)", value=float(raw_defaults["S5_CO2"]),
                min_value=0.0, step=10.0, format="%.1f"
            )
        if groups["co2_slope"]:
            raw["S5_CO2_Slope"] = st.number_input(
                "CO₂ slope S5", value=float(raw_defaults["S5_CO2_Slope"]),
                step=0.1, format="%.4f"
            )
        if groups["pir"]:
            pir_cols = st.columns(2)
            raw["S6_PIR"] = pir_cols[0].selectbox(
                "S6 PIR", [0, 1], index=int(round(raw_defaults["S6_PIR"]))
            )
            raw["S7_PIR"] = pir_cols[1].selectbox(
                "S7 PIR", [0, 1], index=int(round(raw_defaults["S7_PIR"]))
            )

        # Sensor yang tidak dipakai tetap tersedia untuk kelengkapan pencatatan,
        # tetapi sengaja dipisahkan agar tidak memberi kesan memengaruhi prediksi.
        unused_groups = []
        if not groups["temperature"]:
            unused_groups.append(("Temperatur S1-S4 (°C)", [f"S{i}_Temp" for i in range(1, 5)], 0.1, "%.2f"))
        if not groups["light"]:
            unused_groups.append(("Cahaya S1-S4 (lux)", [f"S{i}_Light" for i in range(1, 5)], 1.0, "%.1f"))
        if not groups["sound"]:
            unused_groups.append(("Suara S1-S4 (volt)", [f"S{i}_Sound" for i in range(1, 5)], 0.01, "%.3f"))
        if not groups["co2"]:
            unused_groups.append(("CO₂ S5 (ppm)", ["S5_CO2"], 10.0, "%.1f"))
        if not groups["co2_slope"]:
            unused_groups.append(("CO₂ slope S5", ["S5_CO2_Slope"], 0.1, "%.4f"))
        if not groups["pir"]:
            unused_groups.append(("Sensor PIR", ["S6_PIR", "S7_PIR"], 1.0, "%.0f"))

        if unused_groups:
            with st.expander("Sensor tambahan (tidak digunakan oleh model final)", expanded=False):
                st.info(
                    "Sensor di bawah ini dapat dicatat, tetapi tidak memengaruhi prediksi karena "
                    "fiturnya tidak termasuk feature set terpilih."
                )
                for title, columns, step, fmt in unused_groups:
                    if columns == ["S6_PIR", "S7_PIR"]:
                        pcols = st.columns(2)
                        raw["S6_PIR"] = pcols[0].selectbox(
                            "S6 PIR (tidak digunakan)", [0, 1],
                            index=int(round(raw_defaults["S6_PIR"]))
                        )
                        raw["S7_PIR"] = pcols[1].selectbox(
                            "S7 PIR (tidak digunakan)", [0, 1],
                            index=int(round(raw_defaults["S7_PIR"]))
                        )
                    else:
                        raw.update(sensor_input_group(
                            title, columns, raw_defaults, step, fmt, active=False
                        ))

        submitted = st.form_submit_button(
            "Hitung estimasi okupansi", type="primary", use_container_width=True
        )

    if submitted:
        result = predict_one(bundle, raw)
        feature_values = result["features"]

        outside = []
        for name, limits in bundle.get("feature_ranges_1_99", {}).items():
            value = float(feature_values[name])
            if value < limits[0] or value > limits[1]:
                outside.append((name, value, limits[0], limits[1]))

        st.divider()
        st.subheader("Hasil estimasi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimasi penghuni", f"{result['prediction_count']} orang")
        m2.metric("Output Sugeno kontinu", f"{result['prediction_raw']:.4f}")
        m3.metric("Output setelah batas 0-3", f"{result['prediction_bounded']:.4f}")

        if result["prediction_raw"] < 0 or result["prediction_raw"] > 3:
            st.warning(
                "Output kontinu berada di luar rentang fisik 0-3. Nilai yang ditampilkan sebagai "
                "jumlah orang menggunakan pembatasan rentang sebelum pembulatan."
            )
        if outside:
            st.warning(
                "Sebagian fitur berada di luar rentang persentil 1-99 data development. "
                "Prediksi dapat merupakan ekstrapolasi."
            )
            st.dataframe(
                pd.DataFrame(outside, columns=["Fitur", "Nilai input", "P1 development", "P99 development"]),
                use_container_width=True,
                hide_index=True,
            )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Aktivasi aturan")
            rule_names = [f"R{i+1}" for i in range(len(result["firing"]))]
            firing_df = pd.DataFrame(
                {"Firing strength": result["firing"]}, index=rule_names
            )
            st.bar_chart(firing_df, use_container_width=True)
        with right:
            st.markdown("#### Kontribusi terhadap output")
            contribution_df = pd.DataFrame(
                {"Kontribusi berbobot": result["contributions"]},
                index=rule_names,
            )
            st.bar_chart(contribution_df, use_container_width=True)

        rule_table = pd.DataFrame({
            "Aturan": rule_names,
            "Firing strength": result["firing"],
            "Output lokal": result["local_outputs"],
            "Kontribusi berbobot": result["contributions"],
        })
        st.markdown("#### Rincian inferensi Sugeno")
        st.dataframe(
            rule_table.style.format({
                "Firing strength": "{:.5f}",
                "Output lokal": "{:.5f}",
                "Kontribusi berbobot": "{:.5f}",
            }).background_gradient(subset=["Firing strength"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Output akhir = jumlah (firing strength aturan × output lokal aturan). "
            "Firing strength telah dinormalisasi sehingga jumlahnya 1."
        )

        st.markdown("#### Fitur yang masuk ke model")
        used_feature_df = pd.DataFrame({
            "Fitur": feature_names,
            "Nilai": [feature_values[name] for name in feature_names],
        })
        st.dataframe(used_feature_df, use_container_width=True, hide_index=True)

# ============================================================
# 9. Prediksi banyak data
# ============================================================
elif page == "Prediksi banyak data":
    st.subheader("Prediksi beberapa pembacaan sekaligus")
    st.write(
        "Kamu tidak harus memasukkan satu observasi setiap kali. Gunakan CSV atau tabel input "
        "untuk memproses banyak baris dalam satu proses."
    )

    template = make_csv_template()
    st.download_button(
        "Unduh template CSV sensor",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="template_input_sensor_occupancy.csv",
        mime="text/csv",
    )

    tab_csv, tab_table = st.tabs(["Upload CSV", "Input melalui tabel"])
    with tab_csv:
        uploaded = st.file_uploader("Pilih file CSV", type=["csv"], key="batch_csv")
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                missing_cols = [col for col in required_columns if col not in input_df.columns]
                if missing_cols:
                    st.error("Kolom sensor wajib untuk model ini belum lengkap: " + ", ".join(missing_cols))
                elif input_df.empty:
                    st.warning("File CSV tidak memiliki baris data.")
                elif input_df[required_columns].isna().any().any():
                    st.error("Masih ada nilai kosong pada kolom sensor yang digunakan model.")
                else:
                    st.success(f"{len(input_df):,} baris data berhasil dibaca.")
                    st.dataframe(input_df.head(10), use_container_width=True)
                    if st.button("Proses seluruh baris", type="primary", key="process_csv"):
                        with st.spinner("Menghitung prediksi untuk seluruh observasi..."):
                            result_df = run_batch_dataframe(input_df)
                        st.session_state["batch_result"] = result_df
            except Exception as exc:
                st.error(f"CSV tidak dapat diproses: {exc}")

    with tab_table:
        st.caption(
            "Tambahkan baris menggunakan tombol + pada tabel. Kolom yang ditampilkan mengikuti "
            "sensor yang dibutuhkan oleh feature set model."
        )
        editor_seed = pd.DataFrame({
            col: pd.Series(dtype="float64") for col in required_columns
        })
        edited_df = st.data_editor(
            editor_seed,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_batch_editor",
            column_config={
                col: st.column_config.NumberColumn(col, format="%.4f")
                for col in required_columns
            },
        )
        if st.button("Proses tabel input", type="primary", key="process_table"):
            if edited_df.empty:
                st.warning("Tambahkan minimal satu baris observasi terlebih dahulu.")
            elif edited_df[required_columns].isna().any().any():
                st.error("Lengkapi seluruh kolom sensor yang digunakan model.")
            else:
                with st.spinner("Menghitung prediksi..."):
                    st.session_state["batch_result"] = run_batch_dataframe(edited_df)

    if "batch_result" in st.session_state and not st.session_state["batch_result"].empty:
        result_df = st.session_state["batch_result"]
        st.divider()
        st.markdown("#### Ringkasan hasil batch")
        a, b, c = st.columns(3)
        a.metric("Jumlah observasi", f"{len(result_df):,}")
        b.metric("Rata-rata estimasi kontinu", f"{result_df['prediksi_bounded'].mean():.3f}")
        c.metric("Prediksi terbanyak", f"{int(result_df['prediksi_jumlah_penghuni'].mode().iloc[0])} orang")

        st.markdown("#### Distribusi estimasi jumlah penghuni")
        distribution = (
            result_df["prediksi_jumlah_penghuni"]
            .value_counts().reindex([0, 1, 2, 3], fill_value=0)
            .rename_axis("Jumlah penghuni").to_frame("Jumlah observasi")
        )
        st.bar_chart(distribution, use_container_width=True)

        if "Date" in result_df.columns or "Time" in result_df.columns:
            date_part = result_df["Date"].astype(str) if "Date" in result_df.columns else ""
            time_part = result_df["Time"].astype(str) if "Time" in result_df.columns else ""
            if isinstance(date_part, pd.Series) and isinstance(time_part, pd.Series):
                stamps = pd.to_datetime(date_part + " " + time_part, errors="coerce")
            elif isinstance(date_part, pd.Series):
                stamps = pd.to_datetime(date_part, errors="coerce")
            else:
                stamps = pd.to_datetime(time_part, errors="coerce")
            if stamps.notna().any():
                chart_df = result_df.loc[stamps.notna(), ["prediksi_bounded"]].copy()
                chart_df.index = stamps[stamps.notna()]
                st.markdown("#### Perubahan estimasi sepanjang waktu")
                st.line_chart(chart_df, use_container_width=True)

        rule_cols = [c for c in result_df.columns if c.startswith("firing_R")]
        if rule_cols:
            st.markdown("#### Rata-rata aktivasi aturan")
            st.bar_chart(result_df[rule_cols].mean().to_frame("Rata-rata firing strength"))

        st.markdown("#### Tabel hasil prediksi")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh hasil prediksi CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_prediksi_occupancy_fuzzy.csv",
            mime="text/csv",
        )

# ============================================================
# 10. Tentang model
# ============================================================
else:
    st.subheader("Tentang model yang sedang digunakan")
    st.write(
        "Halaman ini merangkum konfigurasi model dan aturan yang diekspor dari notebook. "
        "Model menghasilkan estimasi kontinu melalui inferensi First-Order Sugeno, kemudian "
        "output dibatasi ke rentang okupansi 0-3 dan dibulatkan ke jumlah orang terdekat."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metode clustering", str(bundle.get("selected_method", "GK")).upper())
    c2.metric("Jumlah aturan", str(bundle.get("n_rules", "?")))
    c3.metric("Fuzzifier (m)", str(bundle.get("m", "?")))
    c4.metric("Feature set", str(bundle.get("selected_feature_set", "?")))

    st.markdown("#### Fitur yang digunakan")
    feature_info = pd.DataFrame({
        "Fitur model": feature_names,
        "Keterangan": [
            {
                "temp_mean": "Rata-rata temperatur empat sensor",
                "temp_std": "Variasi temperatur empat sensor",
                "light_mean_log": "Log(1 + rata-rata intensitas cahaya S1–S4)",
                "light_std_log": "Log(1 + simpangan baku cahaya S1–S4)",
                "sound_mean_log": "Log(1 + rata-rata suara S1–S4)",
                "sound_std_log": "Log(1 + simpangan baku suara S1–S4)",
                "co2_log": "Log(1 + konsentrasi CO₂ S5)",
                "co2_ppm": "Konsentrasi CO₂ S5 dalam ppm",
                "co2_slope": "Kemiringan perubahan CO₂ S5",
                "pir_active_count": "Jumlah sensor PIR aktif dari S6 dan S7",
            }.get(name, "Fitur hasil feature engineering")
            for name in feature_names
        ],
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    st.markdown("#### Basis aturan")
    if "rule_base_table" in bundle:
        rule_base = bundle["rule_base_table"]
        if isinstance(rule_base, pd.DataFrame):
            st.dataframe(rule_base, use_container_width=True, hide_index=True)
        else:
            st.write(rule_base)
    else:
        st.info("Tabel rule base tidak ditemukan di dalam bundle.")

    st.markdown("#### Profil cluster")
    if "rule_profile" in bundle:
        profile = bundle["rule_profile"]
        if isinstance(profile, pd.DataFrame):
            st.dataframe(profile, use_container_width=True, hide_index=True)
        else:
            st.write(profile)

    st.markdown("#### Catatan penggunaan")
    st.info(
        "GUI ini adalah demonstrator inferensi model yang telah dilatih. GUI tidak melatih ulang "
        "model dan tidak menjamin generalisasi ke ruangan, sensor, atau kondisi lingkungan lain."
    )