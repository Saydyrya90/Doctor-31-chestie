import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# --- Import your anomaly detection module's functions ---
try:
    from anomaly_detector import (
        get_anomaly_detection_pipeline,
        initialize_anomaly_columns,
        finalize_anomaly_data,
        SEVERITY_LEVEL_MAP,
        RULE_DEFINITIONS
    )
except ImportError:
    st.error("Failed to import from 'anomaly_detector'. Make sure 'anomaly_detector.py' is in the 'src' directory.")
    st.stop()

# --- Configuration & Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'doctor31_cazuri(1).csv')

st.set_page_config(layout="wide")
st.title("Doctor31 Data Anomaly Detection (Modular with Progress & Severity)")

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
    numeric_cols_check = ['age_v', 'greutate', 'inaltime', 'imcINdex']
    for col in numeric_cols_check:
        if col in df_orig.columns:
            df_orig[col] = pd.to_numeric(df_orig[col], errors='coerce')
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
    df_to_process = initialize_anomaly_columns(df_to_process)

    numeric_cols_check = ['age_v', 'greutate', 'inaltime', 'imcINdex']
    for col in numeric_cols_check:
        if col in df_to_process.columns:
            df_to_process[col] = pd.to_numeric(df_to_process[col], errors='coerce')
        else:
            st.warning(f"Column '{col}' not found for numeric conversion. Rules relying on it might fail.")

    rules_pipeline = get_anomaly_detection_pipeline()
    total_steps = len(rules_pipeline)

    for i, (message, rule_function) in enumerate(rules_pipeline):
        status_text.text(f"{message}...")
        try:
            df_to_process = rule_function(df_to_process)
        except Exception as e:
            status_text.error(f"Error during '{message}': {e}")
            st.exception(e)
            st.stop()
        progress_bar.progress((i + 1) / total_steps)

    df_with_anomalies = finalize_anomaly_data(df_to_process)
    status_text.success("Anomaly detection complete!")

    # --- Reporting in Streamlit ---
    anomalous_df_st = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()

    if 'anomaly_reason' in anomalous_df_st.columns:
        anomalous_df_st['anomaly_reason_str'] = anomalous_df_st['anomaly_reason'].apply(
            lambda x: '; '.join(x) if isinstance(x, list) and x else "N/A"
        )
    else:
        anomalous_df_st['anomaly_reason_str'] = "Reason processing error"

    if not anomalous_df_st.empty and 'max_anomaly_severity_value' in anomalous_df_st.columns and 'anomaly_score_percentage' in anomalous_df_st.columns:
        anomalous_df_st = anomalous_df_st.sort_values(
            by=['max_anomaly_severity_value', 'anomaly_score_percentage'],
            ascending=[False, False]
        )

    # --- Display Anomaly Reason Counts ---
    if not anomalous_df_st.empty and 'applied_rule_codes' in anomalous_df_st.columns:
        all_rule_codes_flat = []
        for code_list in anomalous_df_st['applied_rule_codes']:
            if isinstance(code_list, list):
                all_rule_codes_flat.extend(code_list)
        
        if all_rule_codes_flat:
            reason_code_counts = pd.Series(all_rule_codes_flat).value_counts()
            reason_display_counts = reason_code_counts.rename(index=lambda rc: f"{rc}: {RULE_DEFINITIONS.get(rc, ('Unknown Rule', '', 0))[0]}")
            st.subheader("Top Anomaly Reasons Counts:")
            st.dataframe(reason_display_counts)
        else:
            st.info("No specific anomaly rule codes were recorded.")

    st.subheader(f"Anomalous Rows Found: {len(anomalous_df_st)}")

    # --- UPDATED LEGEND ---
    if not anomalous_df_st.empty:
        st.markdown("""
        **Legend for Anomaly Severity:**
        - <span style='background-color:#FFCCCB; color:black; padding: 2px 5px; border-radius: 3px;'>Red</span>: Clearly anomalous, data errors, or impossible values.
          > *Example: Age > 120 years (R1.1), BMI > 70 (R4.2), Height < 50cm (R3.1).*
        - <span style='background-color:#FFD580; color:black; padding: 2px 5px; border-radius: 3px;'>Orange</span>: Suspicious data, inconsistencies, or potential process/data management issues.
          > *Example: Elderly (Age > 85) and Obese (R5.1), Potential duplicate entry (R6.1), Calculated BMI differs from provided (R7.1).*
        - <span style='background-color:#FFFFE0; color:black; padding: 2px 5px; border-radius: 3px;'>Yellow</span>: Lower severity warnings or minor deviations (if rules defined).
          > *Example: Currently no 'Yellow' rules are defined, but could include minor statistical deviations.*
        """, unsafe_allow_html=True)
        st.markdown("---")

    if not anomalous_df_st.empty:
        display_cols_st = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'imcINdex',
                           'max_anomaly_severity_category', 'anomaly_score_percentage',
                           'data1', 'anomaly_reason_str']
        if 'calculated_bmi' in anomalous_df_st.columns:
            if 'data1' in display_cols_st:
                data1_idx = display_cols_st.index('data1')
                display_cols_st.insert(data1_idx, 'calculated_bmi')
            else:
                display_cols_st.append('calculated_bmi')

        actual_display_cols = [col for col in display_cols_st if col in anomalous_df_st.columns]
        missing_cols = [col for col in display_cols_st if col not in anomalous_df_st.columns]
        if missing_cols:
            st.warning(f"The following columns were expected for display but not found in anomalous data: {', '.join(missing_cols)}")
        
        # Styling function for rows (text color is now black)
        def highlight_severity(row):
            severity = row.get('max_anomaly_severity_category', '')
            text_color = 'black' # Ensure text is black
            bg_color = 'white'   # Default background

            if severity == 'Red': bg_color = '#FFCCCB' # Light red
            elif severity == 'Orange': bg_color = '#FFD580' # Light orange
            elif severity == 'Yellow': bg_color = '#FFFFE0' # Light yellow
            return [f'background-color: {bg_color}; color: {text_color}' for _ in row]

        if actual_display_cols:
            st.dataframe(anomalous_df_st[actual_display_cols].style.apply(highlight_severity, axis=1))
        else:
            st.warning("No columns available to display for anomalous data.")

        @st.cache_data
        def convert_df_to_csv(input_df_to_convert, columns_to_export):
            if not columns_to_export: return "".encode('utf-8')
            valid_cols_for_export = [col for col in columns_to_export if col in input_df_to_convert.columns]
            if not valid_cols_for_export: return "".encode('utf-8')
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
        'is_anomaly', 'anomaly_reason', 'applied_rule_codes', 'total_anomaly_score',
        'max_anomaly_severity_value', 'max_anomaly_severity_category', 'anomaly_score_percentage',
        'imcINdex_cleaned', 'inaltime_m', 'calculated_bmi',
        'prev_data1', 'time_diff_to_prev_hours'
    ]
    existing_cols_to_drop_valid = [col for col in cols_to_drop_from_valid if col in valid_df_st.columns]
    if existing_cols_to_drop_valid:
        valid_df_st.drop(columns=existing_cols_to_drop_valid, inplace=True, errors='ignore')

    st.subheader(f"Valid Rows (First 5 Rows): {len(valid_df_st)}")
    st.dataframe(valid_df_st.head())

    if not valid_df_st.empty:
        csv_valid = valid_df_st.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Valid Data as CSV",
            data=csv_valid,
            file_name='valid_cases_streamlit.csv',
            mime='text/csv',
        )
else:
    st.error("Could not load or process the data. Please check the data file path and format.")