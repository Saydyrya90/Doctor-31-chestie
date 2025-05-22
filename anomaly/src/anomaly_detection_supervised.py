import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etichetare automată a unor cazuri: 1 = anomalie, 0 = normal.
    Folosește reguli simple pentru antrenare supervizată.
    """
    df = df.copy()
    df["label"] = None

    # Cazuri normale
    normal_cond = (
        (df["bmi"] >= 18.5) & (df["bmi"] <= 25) &
        (df["age"] >= 20) & (df["age"] <= 60) &
        (df["weight"] >= 55) & (df["weight"] <= 90)
    )
    df.loc[normal_cond, "label"] = 0

    # Cazuri anormale
    anomaly_cond = (
        (df["bmi"] < 13) | (df["bmi"] > 55) |
        ((df["age"] > 85) & (df["bmi"] > 30)) |
        (df["weight"] < 30) | (df["weight"] > 160)
    )
    df.loc[anomaly_cond, "label"] = 1

    # Păstrăm doar cazurile etichetate (nu toate au label)
    return df[df["label"].notna()]


# Model vechi RandomForest
# def train_supervised_anomaly_model(df_labeled: pd.DataFrame):
    """
    Antrenează un model supervizat pe datele etichetate.
    """
#    features = ["age", "weight", "height", "bmi"]

    # Eliminăm valorile inf/NaN
#    X = df_labeled[features].replace([float('inf'), float('-inf')], pd.NA).dropna()
#    y = df_labeled.loc[X.index, "label"].astype(int)

#    scaler = StandardScaler()
#    X_scaled = scaler.fit_transform(X)

#    model = RandomForestClassifier(n_estimators=100, random_state=42)
#    model.fit(X_scaled, y)


#    return model, scaler

# Model nou XGBoost

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def train_supervised_anomaly_model(df_labeled):
    features = ["age", "weight", "height", "bmi"]
    
    # Preprocesare
    X = df_labeled[features].replace([float('inf'), float('-inf')], pd.NA).dropna()
    y = df_labeled.loc[X.index, "label"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model îmbunătățit cu XGBoost
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)

    return model, scaler




def apply_supervised_model(df: pd.DataFrame, model, scaler):
    """
    Aplică modelul supervizat pe toate datele și returnează scorurile.
    """
    df = df.copy()
    features = ["age", "weight", "height", "bmi"]
    X = df[features].replace([float('inf'), float('-inf')], pd.NA).fillna(0)
    X_scaled = scaler.transform(X)

    # predict_proba → scor de anomalie = probabilitate ca label = 1
    scores = model.predict_proba(X_scaled)[:, 1]
    df["ai_anomaly_score"] = model.predict_proba(X_scaled)[:, 1]

    return df


if __name__ == "__main__":
    from data_loader import load_data_from_pdf
    from validators import validate_data

    df = load_data_from_pdf("dataset.pdf")
    df = validate_data(df)

    # Etichetăm un subset
    df_labeled = generate_labels(df)

    print(f"📊 Cazuri etichetate automat: {len(df_labeled)}")

    # Antrenăm modelul
    model, scaler = train_supervised_anomaly_model(df_labeled)

    # Aplicăm pe tot setul
    df = apply_supervised_model(df, model, scaler)

    # Afișăm top 5 scoruri
    print("\n🔴 Top cazuri cu scor de anomalie mare:")
    print(df.sort_values("ai_anomaly_score", ascending=False)[["age", "weight", "height", "bmi", "ai_anomaly_score"]].head())

    # Export (opțional)
    df.to_csv("export_supervised_anomalies.csv", index=False)
    print("\n💾 Scoruri salvate în export_supervised_anomalies.csv")

# Salvăm modelul și scalerul antrenat
joblib.dump(model, "model_supervised.pkl")
joblib.dump(scaler, "scaler_supervised.pkl")

print("✅ Modelul și scalerul au fost salvați cu succes.")

