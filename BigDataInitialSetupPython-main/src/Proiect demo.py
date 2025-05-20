import streamlit as st
import pandas as pd
import numpy as np
import os
import time # For simulating progress, can be removed if individual steps are slow enough

# --- Import your anomaly detection module's functions ---
try:
    from anomaly_detector import (
        apply_age_rules,
        apply_weight_rules,
        apply_height_rules,
        apply_bmi_rules,
        apply_elderly_obesity_rule,
        apply_duplicate_case_rule,
        apply_bmi_consistency_rule
    )
except ImportError:
    st.error("Failed to import from 'anomaly_detector'. Make sure 'anomaly_detector.py' is in the 'src' directory.")
    st.stop()

# --- Configuration & Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'doctor31_cazuri(1).csv')

st.set_page_config(layout="wide")
st.title("Doctor31 Data Anomaly Detection (Modular with Progress)")

# --- Load and Preprocess Data (Function for caching) ---
@st.cache_data
def load_data(file_path):
    try:
        df_orig = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None
    df_orig['data1'] = pd.to_datetime(df_orig['data1'], errors='coerce')
    return df_orig

raw_df = load_data(DATA_FILE)

if raw_df is not None:
    st.subheader("Original Data Sample (First 5 Rows)")
    st.dataframe(raw_df.head())

    # --- Detect Anomalies with Progress Bar ---
    st.subheader("Anomaly Detection Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    df_to_process = raw_df.copy()

    if 'is_anomaly' not in df_to_process.columns:
        df_to_process['is_anomaly'] = False
    if 'anomaly_reason' not in df_to_process.columns:
        df_to_process['anomaly_reason'] = [[] for _ in range(len(df_to_process))]

    numeric_cols_check = ['age_v', 'greutate', 'inaltime', 'imcINdex']
    for col in numeric_cols_check:
        if col in df_to_process.columns:
            df_to_process[col] = pd.to_numeric(df_to_process[col], errors='coerce')
        else:
            st.warning(f"Column '{col}' not found for numeric conversion. Rules relying on it might fail.")

    rules_to_apply = [
        ("Applying Age Rules", apply_age_rules),
        ("Applying Weight Rules", apply_weight_rules),
        ("Applying Height Rules", apply_height_rules),
        ("Applying BMI Rules (from imcINdex)", apply_bmi_rules),
        ("Applying Elderly Obesity Rule", apply_elderly_obesity_rule),
        ("Applying Duplicate Case Rule", apply_duplicate_case_rule),
        ("Applying BMI Consistency Rule", apply_bmi_consistency_rule)
    ]
    total_steps = len(rules_to_apply)

    for i, (message, rule_function) in enumerate(rules_to_apply):
        status_text.text(f"{message}...")
        try:
            df_to_process = rule_function(df_to_process.copy())
        except Exception as e:
            status_text.error(f"Error during '{message}': {e}")
            st.exception(e) # Also print full traceback to Streamlit for debugging
            st.stop()
        progress_bar.progress((i + 1) / total_steps)
        # time.sleep(0.1) # Optional

    status_text.success("Anomaly detection complete!")
    df_with_anomalies = df_to_process

    # --- Reporting in Streamlit ---
    anomalous_df_st = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()

    if 'anomaly_reason' in anomalous_df_st.columns:
        anomalous_df_st['anomaly_reason_str'] = anomalous_df_st['anomaly_reason'].apply(
            lambda x: '; '.join(x) if isinstance(x, list) and x else "N/A"
        )
    else:
        anomalous_df_st['anomaly_reason_str'] = "Reason processing error"

    # --- Display Anomaly Reason Counts ---
    if not anomalous_df_st.empty and 'anomaly_reason' in anomalous_df_st.columns:
        all_reasons_flat = []
        for reason_list in anomalous_df_st['anomaly_reason']: # Use the list version from df_with_anomalies
            if isinstance(reason_list, list):
                all_reasons_flat.extend(reason_list)
        
        if all_reasons_flat:
            reason_counts = pd.Series(all_reasons_flat).value_counts()
            st.subheader("Top Anomaly Reasons Counts:")
            st.dataframe(reason_counts)
        else:
            st.info("No specific anomaly reasons were recorded (anomaly_reason list was empty for all anomalous rows).")

    st.subheader(f"Anomalous Rows Found: {len(anomalous_df_st)}")

    if not anomalous_df_st.empty:
        display_cols_st = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'imcINdex', 'data1', 'anomaly_reason_str']
        if 'calculated_bmi' in anomalous_df_st.columns:
            if 'data1' in display_cols_st:
                data1_index = display_cols_st.index('data1')
                display_cols_st.insert(data1_index, 'calculated_bmi')
            else:
                display_cols_st.append('calculated_bmi')

        actual_display_cols = [col for col in display_cols_st if col in anomalous_df_st.columns]
        missing_cols = [col for col in display_cols_st if col not in anomalous_df_st.columns]
        if missing_cols:
            st.warning(f"The following columns were expected for display but not found in anomalous data: {', '.join(missing_cols)}")
        if actual_display_cols:
            st.dataframe(anomalous_df_st[actual_display_cols])
        else:
            st.warning("No columns available to display for anomalous data.")

        @st.cache_data
        def convert_df_to_csv(input_df_to_convert, columns_to_export):
            # Ensure only existing columns are passed to to_csv
            if not columns_to_export: # If actual_display_cols was empty
                return "".encode('utf-8')
            valid_cols_for_export = [col for col in columns_to_export if col in input_df_to_convert.columns]
            if not valid_cols_for_export:
                return "".encode('utf-8')
            return input_df_to_convert[valid_cols_for_export].to_csv(index=False).encode('utf-8')

        if not anomalous_df_st.empty and actual_display_cols:
            csv_anomalous = convert_df_to_csv(anomalous_df_st, actual_display_cols)
            st.download_button(
                label="Download Anomalous Data as CSV",
                data=csv_anomalous,
                file_name='anomalous_cases_report_streamlit.csv',
                mime='text/csv',
            )
    else:
        st.success("No anomalies found based on the defined rules.")

    valid_df_st = df_with_anomalies[~df_with_anomalies['is_anomaly']].copy()
    cols_to_drop_from_valid = [
        'is_anomaly', 'anomaly_reason', 'imcINdex_cleaned',
        'inaltime_m', 'calculated_bmi',
        'prev_data1', 'time_diff_to_prev_hours'
    ]
    existing_cols_to_drop_valid = [col for col in cols_to_drop_from_valid if col in valid_df_st.columns]
    if existing_cols_to_drop_valid:
        valid_df_st.drop(columns=existing_cols_to_drop_valid, inplace=True, errors='ignore')

    st.subheader(f"Valid Rows (First 5 Rows): {len(valid_df_st)}")
    st.dataframe(valid_df_st.head())

    if not valid_df_st.empty:
        # For valid_df_st, we can directly use its current columns for CSV export
        csv_valid = valid_df_st.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Valid Data as CSV",
            data=csv_valid,
            file_name='valid_cases_streamlit.csv',
            mime='text/csv',
        )
else:
    st.error("Could not load or process the data. Please check the data file path and format.")