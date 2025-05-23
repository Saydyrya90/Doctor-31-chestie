import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import time
import shap

# --- Project-Specific Imports (Corrected) ---
try:
    # Correctly import specific names from rule_based_detector
    from src.rule_based_detector import (
        get_anomaly_detection_pipeline,
        initialize_anomaly_columns,
        finalize_anomaly_data,
        RULE_DEFINITIONS,
        SEVERITY_LEVEL_MAP # Ensure this is defined in rule_based_detector.py if it's used here directly
                           # (or access it as rule_based_detector.SEVERITY_LEVEL_MAP if not imported directly)
    )
except ImportError as e:
    st.error(f"Error importing from src.rule_based_detector: {e}. Ensure 'src/rule_based_detector.py' exists and src/ contains an __init__.py file.")
    st.stop()

try:
    from src.data_loader import load_and_prepare_data, COLUMN_MAPPING_TO_COMMON
    from src.validators import validate_data as validate_data_for_ai
    from src.anomaly_detection_supervised import (
        generate_labels as generate_labels_supervised,
        train_supervised_anomaly_model,
        apply_supervised_model
    )
    from src.explain import explain_instance
except ImportError as e:
    st.error(f"Error importing new system components: {e}. Ensure all necessary files are in src/ and src/ contains an __init__.py file.")
    st.stop()


# --- Global Configuration ---
st.set_page_config(layout="wide")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # streamlit_app.py is at project root
DATA_FILE_ORIGINAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'doctor31_cazuri(1).csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SUPERVISED_MODEL_PATH = os.path.join(MODEL_DIR, "model_supervised_doctor31.pkl")
SUPERVISED_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_supervised_doctor31.pkl")


@st.cache_data
def load_initial_data(file_path):
    df_orig, df_comm = load_and_prepare_data(file_path)
    if df_orig is None:
        st.error(f"Failed to load data from {file_path}")
    return df_orig, df_comm

@st.cache_resource
def get_or_train_ai_model_cached(df_common_cols_for_training):
    if os.path.exists(SUPERVISED_MODEL_PATH) and os.path.exists(SUPERVISED_SCALER_PATH):
        st.info("Loading pre-trained supervised model and scaler...")
        model = joblib.load(SUPERVISED_MODEL_PATH)
        scaler = joblib.load(SUPERVISED_SCALER_PATH)
        st.success("Pre-trained model and scaler loaded.")
        return model, scaler
    else:
        st.info("No pre-trained model found. Training a new supervised model for Doctor31 data...")
        with st.spinner("Generating labels for AI model training..."):
            required_cols_for_labels = ['age', 'weight', 'height', 'bmi']
            if not all(col in df_common_cols_for_training.columns and df_common_cols_for_training[col].notna().any() for col in required_cols_for_labels):
                missing_or_empty = [col for col in required_cols_for_labels if col not in df_common_cols_for_training.columns or not df_common_cols_for_training[col].notna().any()]
                st.error(f"Missing or empty crucial columns for label generation: {missing_or_empty}. Cannot train AI model.")
                return None, None
            
            df_for_labeling = df_common_cols_for_training.dropna(subset=required_cols_for_labels).copy()
            if df_for_labeling.empty:
                st.error("No data left for labeling after dropping NaNs in essential features.")
                return None, None

            df_labeled = generate_labels_supervised(df_for_labeling)
        
        if df_labeled.empty or len(df_labeled["label"].dropna()) < 20 :
            st.warning(f"Not enough labeled data generated ({len(df_labeled['label'].dropna())} usable labels) to train a reliable supervised model. Need at least 20.")
            return None, None
        if df_labeled["label"].nunique() < 2:
            st.warning(f"Only one class present in generated labels. Cannot train supervised model. Label distribution:\n{df_labeled['label'].value_counts()}")
            return None, None

        with st.spinner("Training supervised AI anomaly model (XGBoost)... This may take a moment."):
            try:
                model, scaler = train_supervised_anomaly_model(df_labeled)
                joblib.dump(model, SUPERVISED_MODEL_PATH)
                joblib.dump(scaler, SUPERVISED_SCALER_PATH)
                st.success("New supervised model trained and saved.")
                return model, scaler
            except Exception as e_train:
                st.error(f"Error during model training: {e_train}")
                st.exception(e_train)
                return None, None

def highlight_severity_rule_based(row):
    severity_category = row.get('max_anomaly_severity_category', '')
    text_color = 'black'
    bg_color = 'white'
    if severity_category == 'Red': bg_color = '#FFCCCB'
    elif severity_category == 'Orange': bg_color = '#FFD580'
    elif severity_category == 'Yellow': bg_color = '#FFFFE0'
    return [f'background-color: {bg_color}; color: {text_color}' for _ in row]

def get_ai_risk_color_text(score):
    if score > 0.9: return "Red"
    elif score > 0.5: return "Orange"
    elif score > 0.2: return "Yellow"
    else: return "Green"

# --- Main App ---
st.title("Doctor31 Data Anomaly Detection Platform")

df_original_cols, df_common_cols = load_initial_data(DATA_FILE_ORIGINAL_CSV)

if df_original_cols is not None and df_common_cols is not None:
    st.sidebar.header("⚙️ Detection Mode")
    detection_mode = st.sidebar.radio(
        "Choose Anomaly Detection Approach:",
        ("Rule-Based (Severity Scoring)", "AI Supervised (XGBoost)")
    )
    st.sidebar.markdown("---")

    st.subheader("Original Data Sample (First 5 Rows)")
    st.dataframe(df_original_cols.head())
    st.markdown("---")

    if detection_mode == "Rule-Based (Severity Scoring)":
        st.header("Rule-Based Anomaly Detection Results")
        df_processed_rb = df_original_cols.copy()
        # Now these calls should work directly because the names are imported
        df_processed_rb = initialize_anomaly_columns(df_processed_rb)

        st.subheader("Detection Progress")
        progress_bar_rb = st.progress(0)
        status_text_rb = st.empty()
        rules_pipeline_rb = get_anomaly_detection_pipeline()
        total_steps_rb = len(rules_pipeline_rb)

        for i, (message, rule_func) in enumerate(rules_pipeline_rb):
            status_text_rb.text(f"{message}...")
            try:
                df_processed_rb = rule_func(df_processed_rb)
            except Exception as e:
                status_text_rb.error(f"Error during '{message}': {e}")
                st.exception(e); st.stop()
            progress_bar_rb.progress((i + 1) / total_steps_rb)
        
        df_final_rb = finalize_anomaly_data(df_processed_rb)
        status_text_rb.success("Rule-based anomaly detection complete!")

        anomalous_df_rb = df_final_rb[df_final_rb['is_anomaly']].copy()
        if 'anomaly_reason' in anomalous_df_rb.columns:
            anomalous_df_rb['anomaly_reason_str'] = anomalous_df_rb['anomaly_reason'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) and x else "N/A")
        else:
            anomalous_df_rb['anomaly_reason_str'] = "Reason processing error"

        if not anomalous_df_rb.empty and 'max_anomaly_severity_value' in anomalous_df_rb.columns and 'anomaly_score_percentage' in anomalous_df_rb.columns:
            anomalous_df_rb = anomalous_df_rb.sort_values(
                by=['max_anomaly_severity_value', 'anomaly_score_percentage'], ascending=[False, False])

        if not anomalous_df_rb.empty and 'applied_rule_codes' in anomalous_df_rb.columns:
            all_codes = [code for sublist in anomalous_df_rb['applied_rule_codes'] for code in sublist if isinstance(sublist, list)]
            if all_codes:
                counts = pd.Series(all_codes).value_counts().rename(index=lambda rc: f"{rc}: {RULE_DEFINITIONS.get(rc, ('Unknown Rule', '', 0))[0]}")
                st.subheader("Top Anomaly Reasons Counts (Rule-Based):")
                st.dataframe(counts)
            
        st.subheader(f"Anomalous Rows Found (Rule-Based): {len(anomalous_df_rb)}")
        if not anomalous_df_rb.empty:
            st.markdown("""
            **Legend:** <span style='background-color:#FFCCCB;color:black;padding:2px 5px;border-radius:3px;'>Red</span> (High Severity/Error),
            <span style='background-color:#FFD580;color:black;padding:2px 5px;border-radius:3px;'>Orange</span> (Suspicious/Inconsistent),
            <span style='background-color:#FFFFE0;color:black;padding:2px 5px;border-radius:3px;'>Yellow</span> (Minor Warning)
            """, unsafe_allow_html=True)
            st.markdown("---")

            display_cols_rb = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'imcINdex',
                               'max_anomaly_severity_category', 'anomaly_score_percentage',
                               'data1', 'anomaly_reason_str']
            if 'rule_based_calculated_bmi' in anomalous_df_rb.columns:
                if 'data1' in display_cols_rb:
                    display_cols_rb.insert(display_cols_rb.index('data1'), 'rule_based_calculated_bmi')
                else:
                    display_cols_rb.append('rule_based_calculated_bmi')
            
            actual_cols_rb = [col for col in display_cols_rb if col in anomalous_df_rb.columns]
            rows_to_show_rb = 1000
            show_all_rb = st.checkbox("Show all anomalous rows (Rule-Based)?", value=len(anomalous_df_rb) <= rows_to_show_rb, key="show_all_rb_checkbox")
            display_subset_rb = anomalous_df_rb if show_all_rb else anomalous_df_rb.head(rows_to_show_rb)
            if not show_all_rb and len(anomalous_df_rb) > rows_to_show_rb:
                st.caption(f"Displaying first {rows_to_show_rb} of {len(anomalous_df_rb)} anomalous rows.")
            if actual_cols_rb:
                st.dataframe(display_subset_rb[actual_cols_rb].style.apply(highlight_severity_rule_based, axis=1))
            
            @st.cache_data
            def convert_df_to_csv_r(input_df, cols): return input_df[cols].to_csv(index=False).encode('utf-8')
            csv_anomalous_rb_dl = convert_df_to_csv_r(anomalous_df_rb, actual_cols_rb)
            st.download_button("Download ALL Anomalous Data (Rule-Based)", csv_anomalous_rb_dl, "anomalies_rule_based.csv", "text/csv", key="download_anom_rb")

        valid_df_rb = df_final_rb[~df_final_rb['is_anomaly']].copy()
        cols_to_drop_rb = list(set(valid_df_rb.columns) - set(df_original_cols.columns))
        valid_df_rb.drop(columns=cols_to_drop_rb, inplace=True, errors='ignore')
        st.subheader(f"Valid Rows (Rule-Based) - First 5: {len(valid_df_rb)}")
        st.dataframe(valid_df_rb.head())
        if not valid_df_rb.empty:
            csv_valid_rb_dl = valid_df_rb.to_csv(index=False).encode('utf-8')
            st.download_button("Download Valid Data (Rule-Based)", csv_valid_rb_dl, "valid_rule_based.csv", "text/csv", key="download_valid_rb")

    elif detection_mode == "AI Supervised (XGBoost)":
        st.header("AI Supervised Anomaly Detection (XGBoost)")
        df_ai_input = df_common_cols.copy()
        df_validated_for_ai = validate_data_for_ai(df_ai_input)
        st.write(f"Data validated for AI: {len(df_validated_for_ai[df_validated_for_ai['valid']])} valid rows, {len(df_validated_for_ai[~df_validated_for_ai['valid']])} invalid rows (by src.validators).")
        
        model_ai, scaler_ai = get_or_train_ai_model_cached(df_validated_for_ai)

        if model_ai and scaler_ai:
            with st.spinner("Applying supervised model to all data..."):
                df_ai_processed = apply_supervised_model(df_validated_for_ai.copy(), model_ai, scaler_ai)
            df_ai_processed["anomaly_risk_category_text"] = df_ai_processed["ai_anomaly_score"].apply(get_ai_risk_color_text)

            st.sidebar.header("🔍 AI Mode Filtering")
            min_age_ai = int(df_ai_processed["age"].min(skipna=True)) if df_ai_processed["age"].notna().any() else 0
            max_age_ai = int(df_ai_processed["age"].max(skipna=True)) if df_ai_processed["age"].notna().any() else 120
            min_age_input_ai = st.sidebar.number_input("Vârstă minimă:", min_value=min_age_ai, max_value=max_age_ai, value=min_age_ai, step=1, key="age_min_ai")
            max_age_input_ai = st.sidebar.number_input("Vârstă maximă:", min_value=min_age_ai, max_value=max_age_ai, value=max_age_ai, step=1, key="age_max_ai")
            sex_options_ai = ["Toate"] + df_ai_processed["sex"].dropna().unique().tolist()
            selected_sex_ai = st.sidebar.selectbox("Sex:", sex_options_ai, key="sex_ai")
            bmi_valid_ai = df_ai_processed["bmi"].replace([np.inf, -np.inf], np.nan).dropna()
            min_bmi_ai = float(bmi_valid_ai.min()) if not bmi_valid_ai.empty else 0.0
            max_bmi_ai = float(bmi_valid_ai.max()) if not bmi_valid_ai.empty else 70.0
            min_bmi_input_ai = st.sidebar.number_input("BMI minim:", min_value=min_bmi_ai, max_value=max_bmi_ai, value=min_bmi_ai, format="%.2f", key="bmi_min_ai")
            max_bmi_input_ai = st.sidebar.number_input("BMI maxim:", min_value=min_bmi_ai, max_value=max_bmi_ai, value=max_bmi_ai, format="%.2f", key="bmi_max_ai")
            min_score_ai = float(df_ai_processed["ai_anomaly_score"].min(skipna=True)) if df_ai_processed["ai_anomaly_score"].notna().any() else 0.0
            max_score_ai = float(df_ai_processed["ai_anomaly_score"].max(skipna=True)) if df_ai_processed["ai_anomaly_score"].notna().any() else 1.0
            default_score_ai = min_score_ai if min_score_ai > 0.01 else 0.01
            score_threshold_ai = st.sidebar.slider("🔎 Scor AI minim:", min_score_ai, max_score_ai, default_score_ai, step=0.01, key="score_ai")

            df_show_ai = df_ai_processed.copy()
            if df_show_ai["age"].notna().any(): df_show_ai = df_show_ai[(df_show_ai["age"] >= min_age_input_ai) & (df_show_ai["age"] <= max_age_input_ai)]
            if df_show_ai["bmi"].notna().any(): df_show_ai = df_show_ai[(df_show_ai["bmi"] >= min_bmi_input_ai) & (df_show_ai["bmi"] <= max_bmi_input_ai)]
            if selected_sex_ai != "Toate" and df_show_ai["sex"].notna().any(): df_show_ai = df_show_ai[df_show_ai["sex"] == selected_sex_ai]
            if df_show_ai["ai_anomaly_score"].notna().any(): df_show_ai = df_show_ai[df_show_ai["ai_anomaly_score"] >= score_threshold_ai]
            df_show_ai = df_show_ai.sort_values(by="ai_anomaly_score", ascending=False)

            st.subheader("📊 Tabel cu Date și Scoruri AI Supervizate")
            cols_to_show_ai = ['id_case', 'age', 'sex', 'weight', 'height', 'bmi_category', 'bmi', 'timestamp', 'ai_anomaly_score', 'anomaly_risk_category_text', 'valid', 'suspect_elderly_obese']
            actual_cols_ai = [col for col in cols_to_show_ai if col in df_show_ai.columns]
            try:
                from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
                gb = GridOptionsBuilder.from_dataframe(df_show_ai[actual_cols_ai])
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                gb.configure_default_column(editable=False, filterable=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
                cell_style_jscode = JsCode("""
                function(params) {
                    if (params.column.colId === 'anomaly_risk_category_text') {
                        if (params.value === 'Red') { return {'backgroundColor': '#FFCCCB', 'color': 'black'}; }
                        else if (params.value === 'Orange') { return {'backgroundColor': '#FFD580', 'color': 'black'}; }
                        else if (params.value === 'Yellow') { return {'backgroundColor': '#FFFFE0', 'color': 'black'}; }
                        else if (params.value === 'Green') { return {'backgroundColor': 'lightgreen', 'color': 'black'}; }
                    } return {'color': 'black'}; }; """)
                gb.configure_columns(actual_cols_ai, cellStyle=cell_style_jscode)
                gridOptions = gb.build()
                AgGrid(df_show_ai[actual_cols_ai], gridOptions=gridOptions, height=600, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, theme='streamlit')
            except ImportError:
                st.warning("streamlit-aggrid not installed. `pip install streamlit-aggrid`")
                if actual_cols_ai: st.dataframe(df_show_ai[actual_cols_ai])

            st.markdown("---")
            st.subheader("🧠 Explică un Caz cu SHAP (Model Supervizat)")
            if not df_show_ai.empty:
                max_idx_shap = len(df_show_ai)-1
                if max_idx_shap >=0:
                    idx_explain_ai = st.number_input("Alege un index de rând din tabelul filtrat:", 0, max_idx_shap, 0, 1, key="shap_idx_ai")
                    if 0 <= idx_explain_ai <= max_idx_shap:
                        case_series_ai = df_show_ai.iloc[idx_explain_ai]
                        features_ai = ["age", "weight", "height", "bmi"]
                        instance_shap = case_series_ai[features_ai].copy()
                        if instance_shap.isna().any():
                            st.warning(f"Instanța SHAP are valori lipsă: {instance_shap[instance_shap.isna()].index.tolist()}. Se înlocuiesc cu 0.")
                            instance_shap.fillna(0, inplace=True)
                        with st.spinner("Generare SHAP..."):
                            try:
                                explainer_s = shap.Explainer(model_ai, feature_names=features_ai)
                                X_selected_scaled_shap = scaler_ai.transform(instance_shap.values.reshape(1, -1))
                                shap_values_s = explainer_s(X_selected_scaled_shap)
                                st.write("Valori intrare SHAP:"); st.dataframe(instance_shap.to_frame().T)
                                st.write("Plot SHAP Waterfall:")
                                fig_s, ax_s = plt.subplots(figsize=(10,4)); shap.plots.waterfall(shap_values_s[0], show=False, max_display=10); st.pyplot(fig_s); plt.close(fig_s)
                            except Exception as e_s: st.error(f"Eroare SHAP: {e_s}")
                    else: st.info("Index SHAP invalid.")
                else: st.info("Tabel filtrat gol pentru SHAP.")
            else: st.error("Model AI sau date procesate lipsesc pentru SHAP.")
        else:
            st.error("Modelul AI nu a putut fi antrenat/încărcat.")
        
        st.markdown("---")
        st.subheader("📊 Diagrame Suplimentare (Mod AI)")
        if model_ai and scaler_ai and 'ai_anomaly_score' in df_ai_processed.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig_score_ai = px.histogram(df_ai_processed, x="ai_anomaly_score", nbins=50, title="Distribuție Scor AI")
                st.plotly_chart(fig_score_ai, use_container_width=True)
            with col2:
                fig_scatter_ai = px.scatter(df_ai_processed, x="age", y="bmi", color="ai_anomaly_score", title="Vârstă vs. BMI (Scor AI)", color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_scatter_ai, use_container_width=True)
else:
    st.error("Datele inițiale nu au putut fi încărcate.")