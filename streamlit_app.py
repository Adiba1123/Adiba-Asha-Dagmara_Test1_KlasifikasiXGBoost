from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

st.set_page_config(page_title="Klasifikasi Cuaca Kolkata", layout="wide")
st.title("Dashboard Klasifikasi Cuaca Kolkata")
st.write("Klasifikasi multikelas `weather_code` menggunakan XGBoost Classifier. Model mengklasifikasikan kondisi cuaca berdasarkan pengukuran hari yang sama, bukan meramalkan cuaca masa depan.")

# Cari dataset yang tersedia di folder aplikasi.
# Nama dataset dapat berupa file asli atau hasil pembersihan.
possible_files = [
    Path("kolkata_weather_cleaned.csv"),
    Path("kolkata_weather_jan_june_2026 (1).csv"),
    Path("kolkata_weather_jan_june_2026.csv"),
]
DATA_PATH = next((p for p in possible_files if p.exists()), None)
if DATA_PATH is None:
    st.error(
        "Dataset CSV tidak ditemukan. Letakkan kolkata_weather_cleaned.csv "
        "atau kolkata_weather_jan_june_2026 (1).csv di folder yang sama "
        "dengan streamlit_app.py."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"], errors="coerce")
data = df.dropna(subset=["weather_code", "time"]).copy()
data["weather_code"] = data["weather_code"].astype(int)
data["year"] = data["time"].dt.year
data["month"] = data["time"].dt.month
data["day_of_year"] = data["time"].dt.dayofyear

feature_cols = [
    c for c in [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "rain_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "wind_direction_10m_dominant", "sunshine_duration",
        "year", "month", "day_of_year"
    ] if c in data.columns
]
X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
y = data["weather_code"]

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah observasi", f"{len(data):,}")
c2.metric("Jumlah fitur", len(feature_cols))
c3.metric("Jumlah kelas", y.nunique())
st.caption(f"Rentang tanggal: {data['time'].min().date()} sampai {data['time'].max().date()}")

with st.expander("Pratinjau dan statistik dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(df.describe(include="all").T, use_container_width=True)
    st.write("Missing values per kolom")
    st.dataframe(df.isna().sum().rename("missing_values"))

st.subheader("Eksplorasi Data (EDA)")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = y.value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title("Proporsi/jumlah kelas weather_code")
    ax.set_xlabel("weather_code")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
with right:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Korelasi fitur")
    st.pyplot(fig)

selected_feature = st.selectbox("Pilih fitur untuk melihat distribusi", feature_cols)
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(X[selected_feature].dropna(), bins=25, kde=True, ax=ax)
ax.set_title(f"Distribusi {selected_feature}")
st.pyplot(fig)

st.subheader("Pelatihan dan evaluasi XGBoost")
test_size = st.slider("Proporsi data uji", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
n_estimators = st.slider("n_estimators", 50, 300, 100, 50)
max_depth = st.slider("max_depth", 2, 10, 5)
learning_rate = st.slider("learning_rate", 0.01, 0.30, 0.10, 0.01)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
imputer = SimpleImputer(strategy="median")
X_train_p = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_p = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

classes = np.sort(y_train.unique())
class_to_int = {label: idx for idx, label in enumerate(classes)}
int_to_class = {idx: label for label, idx in class_to_int.items()}
y_train_enc = y_train.map(class_to_int)
y_test_enc = y_test.map(class_to_int)

model = XGBClassifier(
    n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
    objective="multi:softprob", num_class=len(classes), eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
model.fit(X_train_p, y_train_enc)
pred_enc = model.predict(X_test_p).astype(int)
proba = model.predict_proba(X_test_p)
pred = pd.Series([int_to_class[i] for i in pred_enc], index=y_test.index)

try:
    auc_macro = roc_auc_score(y_test_enc, proba, multi_class="ovr", average="macro",
                              labels=np.arange(len(classes)))
except ValueError:
    auc_macro = np.nan

metrics = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision macro", "Recall macro", "F1-score macro", "ROC-AUC OvR macro"],
    "Nilai": [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average="macro", zero_division=0),
        recall_score(y_test, pred, average="macro", zero_division=0),
        f1_score(y_test, pred, average="macro", zero_division=0),
        auc_macro
    ]
})
st.dataframe(metrics.style.format({"Nilai": "{:.4f}"}), use_container_width=True)
with st.expander("Classification report"):
    st.text(classification_report(y_test, pred, zero_division=0))

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = sorted(pd.unique(y_test).tolist())
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=labels),
                           display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
with right:
    y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, code in enumerate(classes):
        if len(np.unique(y_test_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{code} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve OvR")
    ax.legend(fontsize=7, ncol=2)
    st.pyplot(fig)

st.subheader("Feature importance")
imp = pd.DataFrame({"Fitur": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=imp, x="Importance", y="Fitur", ax=ax)
ax.set_title("Feature Importance XGBoost")
st.pyplot(fig)

st.subheader("Coba prediksi manual")
st.write("Masukkan nilai fitur. Nilai awal diisi menggunakan median dataset.")
with st.form("prediction_form"):
    manual = {}
    medians = X.median(numeric_only=True)
    cols = st.columns(3)
    for i, feature in enumerate(feature_cols):
        with cols[i % 3]:
            val = float(medians[feature]) if pd.notna(medians[feature]) else 0.0
            manual[feature] = st.number_input(feature, value=val, format="%.4f")
    submitted = st.form_submit_button("Prediksi kode cuaca")
if submitted:
    manual_df = pd.DataFrame([manual], columns=feature_cols)
    manual_p = pd.DataFrame(imputer.transform(manual_df), columns=feature_cols)
    p = model.predict_proba(manual_p)[0]
    idx = int(np.argmax(p))
    st.success(f"Prediksi weather_code: {int_to_class[idx]}")
    st.write(f"Probabilitas kelas prediksi: {p[idx]:.2%}")
    st.dataframe(pd.DataFrame({
        "weather_code": [int_to_class[i] for i in range(len(p))],
        "probabilitas": p
    }).sort_values("probabilitas", ascending=False), use_container_width=True)

st.info("Catatan metodologis: pembagian acak 80:20 sesuai praktikum. Untuk mengukur kemampuan memprediksi periode masa depan, gunakan pemisahan kronologis berdasarkan tanggal.")
