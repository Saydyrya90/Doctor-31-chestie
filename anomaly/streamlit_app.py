import streamlit as st
import pandas as pd
import plotly.express as px
import joblib  # pentru a încărca modelul și scalerul
import shap
from src.data_loader import load_data_from_pdf
from src.validators import validate_data
from src.anomaly_detection import detect_anomalies


st.set_page_config(layout="wide")

st.title("🩺 Anomaly Detection în Date Medicale")

# Încarcă și procesează datele - foarte ineficient, datele dupa prima incarcare necesitand din nou incarcare
#with st.spinner("Se încarcă și procesează datele..."):
#    df = load_data_from_pdf("dataset.pdf")
#    df = validate_data(df)
#    df = detect_anomalies(df)

@st.cache_data
def load_and_process():
    model = joblib.load("model_supervised.pkl")
    scaler = joblib.load("scaler_supervised.pkl")

    df = pd.read_csv("export_supervised_anomalies.csv")

    def get_risk_color(score):
        if score > 0.9:
            return "🔴 Mare"
        elif score > 0.5:
            return "🟠 Mediu"
        elif score > 0.2:
            return "🟡 Ușor"
        else:
            return "🟢 OK"
        
    df["anomaly_risk"] = df["ai_anomaly_score"].apply(get_risk_color)
    return df, model, scaler


df, model, scaler = load_and_process()

# Filtrare avansată
st.sidebar.header("🔍 Filtrare avansată")

# --- Vârstă: input numeric minim și maxim ---
min_age = int(df["age"].min())
max_age = int(df["age"].max())

min_age_input = st.sidebar.number_input("Vârstă minimă:", min_value=min_age, max_value=max_age, value=min_age, step=1)
max_age_input = st.sidebar.number_input("Vârstă maximă:", min_value=min_age, max_value=max_age, value=max_age, step=1)

# --- Selectare sex ---
sex_options = df["sex"].dropna().unique().tolist()
selected_sex = st.sidebar.selectbox("Sex:", ["Toate"] + sex_options)

# --- BMI: input numeric minim și maxim ---
bmi_valid = df["bmi"].dropna()
bmi_valid = bmi_valid[(bmi_valid != float('inf')) & (bmi_valid != float('-inf'))]

min_bmi = float(bmi_valid.min())
max_bmi = float(bmi_valid.max())

min_bmi_input = st.sidebar.number_input("BMI minim:", min_value=min_bmi, max_value=max_bmi, value=min_bmi, format="%.2f")
max_bmi_input = st.sidebar.number_input("BMI maxim:", min_value=min_bmi, max_value=max_bmi, value=max_bmi, format="%.2f")

# Checkbox pentru AI anomaly
only_ai_anomalies = st.sidebar.checkbox("✅ Afișează doar anomaliile AI")

# Slider pentru scor AI minim
min_score, max_score = float(df["ai_anomaly_score"].min()), float(df["ai_anomaly_score"].max())
score_threshold = st.sidebar.slider("🔎 Scor AI minim:", min_score, max_score, min_score, step=0.01)

# Selectare tip de date afișate
filt = st.radio(
    "🔍 Ce vrei să vezi?",
    ["Toate datele", "Numai valide", "Numai invalide", "Anomalii AI", "Vârstnici obezi suspecți"],
    horizontal=True
)

if filt == "Numai valide":
    df_show = df[df["valid"]]
elif filt == "Numai invalide":
    df_show = df[~df["valid"]]
elif filt == "Anomalii AI":
    df_show = df[df["ai_anomaly"]]
elif filt == "Vârstnici obezi suspecți":
    df_show = df[df["suspect_elderly_obese"]]
else:
    df_show = df

# 🔽 Aplică filtrele din sidebar asupra datelor afișate
df_show = df_show[
    (df_show["age"] >= min_age_input) & (df_show["age"] <= max_age_input) &
    (df_show["bmi"] >= min_bmi_input) & (df_show["bmi"] <= max_bmi_input)
]


if selected_sex != "Toate":
    df_show = df_show[df_show["sex"] == selected_sex]

# Dacă e bifat "doar anomalii AI", filtrăm pe scor > 0.5 + scor minim
if only_ai_anomalies:
    df_show = df_show[df_show["ai_anomaly_score"] > 0.5]

# Apoi aplicăm întotdeauna scorul minim selectat de utilizator
df_show = df_show[df_show["ai_anomaly_score"] >= score_threshold]



# Afișare tabel
st.subheader("📊 Tabelul cu date și scoruri AI")
st.dataframe(df_show[["age", "weight", "height", "bmi", "ai_anomaly_score", "anomaly_risk"] + [col for col in df_show.columns if col not in ["age", "weight", "height", "bmi", "ai_anomaly_score", "anomaly_risk"]]],
             use_container_width=True)

# Afișare statistici
# ----------------------------------------
# 📉 1. Distribuția BMI
st.subheader("📉 Distribuția valorilor BMI")

fig_bmi = px.histogram(
    df[df["bmi"].notna()],
    x="bmi",
    nbins=40,
    title="Distribuție BMI",
    labels={"bmi": "BMI"},
    color_discrete_sequence=["indianred"]
)
st.plotly_chart(fig_bmi, use_container_width=True)

# ----------------------------------------
# ⚖️ 2. Distribuția greutății
st.subheader("⚖️ Distribuția greutății (kg)")

fig_weight = px.histogram(
    df[df["weight"].notna()],
    x="weight",
    nbins=40,
    title="Distribuție Greutate",
    labels={"weight": "Greutate (kg)"},
    color_discrete_sequence=["royalblue"]
)
st.plotly_chart(fig_weight, use_container_width=True)

# ----------------------------------------
# ✅ 3. Comparativ: valid vs. invalid vs. AI anomaly
st.subheader("📌 Validare vs. Anomalii AI")

df_stats = pd.DataFrame({
    "Categorie": ["Valide", "Invalide", "AI Anomalii"],
    "Număr": [
        df["valid"].sum(),
        (~df["valid"]).sum(),
        (df["ai_anomaly_score"] > 0.5).sum()  # Considerăm scor > 0.5 ca anomalie
    ]
})


fig_compare = px.bar(
    df_stats,
    x="Categorie",
    y="Număr",
    title="Comparativ: Cazuri Valide / Invalide / AI",
    color="Categorie",
    color_discrete_map={
        "Valide": "green",
        "Invalide": "red",
        "AI Anomalii": "orange"
    }
)
st.plotly_chart(fig_compare, use_container_width=True)

# ----------------------------------------
# 🧭 4. Scatter plot Age vs BMI
st.subheader("🧭 Vârstă vs. BMI (colorat după scor AI)")

fig_scatter = px.scatter(
    df[df["bmi"].notna() & df["age"].notna()],
    x="age",
    y="bmi",
    color="anomaly_risk",
    labels={"age": "Vârstă", "bmi": "BMI"},
    title="Distribuție Vârstă vs. BMI (după scor AI)",
    hover_data=["weight", "height", "ai_anomaly_score"]
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ----------------------------------------
# 📏 5. Distribuția înălțimii
st.subheader("📏 Distribuția înălțimii (cm)")

fig_height = px.histogram(
    df[df["height"].notna()],
    x="height",
    nbins=40,
    title="Distribuție Înălțime",
    labels={"height": "Înălțime (cm)"},
    color_discrete_sequence=["purple"]
)
st.plotly_chart(fig_height, use_container_width=True)

# ----------------------------------------
# 🔬 6. Distribuția scorului AI
st.subheader("🔬 Distribuția scorului AI")

fig_score = px.histogram(
    df[df["ai_anomaly_score"].notna()],
    x="ai_anomaly_score",
    nbins=50,
    title="Distribuție Scor AI",
    labels={"ai_anomaly_score": "Scor AI"},
    color_discrete_sequence=["darkorange"]
)
st.plotly_chart(fig_score, use_container_width=True)

from src.explain import explain_instance

# ----------------------------------------
# 🧠 Explică un caz (SHAP)
st.subheader("🧠 Explică un caz (SHAP)")

index_to_explain = st.number_input("Alege un index de caz (rând):", min_value=0, max_value=len(df)-1, value=0, step=1)
selected_case = df.iloc[[index_to_explain]][["age", "weight", "height", "bmi"]]

# Explicație cu SHAP
feature_names = ["age", "weight", "height", "bmi"]

explainer = shap.Explainer(model, feature_names=feature_names)
X_selected_scaled = scaler.transform(selected_case)
shap_values = explainer(X_selected_scaled)


st.write("📋 Valori de intrare:")
st.dataframe(selected_case)

st.write("📊 Explicația scorului AI (SHAP):")

# Afișăm waterfall plot pentru explicarea scorului
shap.plots.waterfall(shap_values[0])

# Afișare în Streamlit
import matplotlib.pyplot as plt
fig = plt.gcf()
st.pyplot(fig)

# ---------------------------------------------------
# 📥 Export date filtrate ca fișier CSV
st.subheader("⬇️ Export date filtrate")

csv = df_show.to_csv(index=False).encode("utf-8")

st.download_button(
    label="💾 Descarcă ca CSV",
    data=csv,
    file_name="date_filtrate.csv",
    mime="text/csv"
)

