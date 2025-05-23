# File: /src/data_loader.py
import pandas as pd
import numpy as np # Make sure numpy is imported
import os

# Define the mapping from your CSV column names (doctor31_cazuri(1).csv)
# to the common names expected by the new AI system's validators.py and anomaly_detection_supervised.py
COLUMN_MAPPING_TO_COMMON = {
    'id_cases': 'id_case', # New system might use 'id_case'
    'age_v': 'age',
    'sex_v': 'sex',
    'agreement': 'agreement', # Keeping as is, can be ignored by AI if not used
    'greutate': 'weight',
    'inaltime': 'height',
    'IMC': 'bmi_category',      # This is the text category
    'data1': 'timestamp',
    'finalizat': 'finalizat',   # Keeping as is
    'testing': 'testing',       # Keeping as is
    'imcINdex': 'bmi'           # This is the numerical BMI value from your CSV
    # Add any other columns from your CSV if they need mapping or should be carried through
}

# Mapping from common names back to your original names (useful if rule_based_detector needs original names)
# This is generated automatically from the above
COLUMN_MAPPING_TO_ORIGINAL = {v: k for k, v in COLUMN_MAPPING_TO_COMMON.items() if k is not None and v is not None}


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Loads data from the primary doctor31_cazuri(1).csv file.
    Performs initial type conversions.
    Returns two DataFrames:
        1. df_original_columns: DataFrame with original column names.
        2. df_common_columns: DataFrame with columns renamed to common names.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        return None, None

    df_original_columns = df.copy()

    # Initial type conversions for original columns
    if 'data1' in df_original_columns.columns:
        df_original_columns['data1'] = pd.to_datetime(df_original_columns['data1'], errors='coerce')
    
    original_numeric_cols = ['age_v', 'greutate', 'inaltime', 'imcINdex']
    for col in original_numeric_cols:
        if col in df_original_columns.columns:
            df_original_columns[col] = pd.to_numeric(df_original_columns[col], errors='coerce')
        else:
            print(f"Warning: Original column '{col}' not found in CSV for initial numeric conversion.")


    # Create DataFrame with common column names
    df_common_columns = df_original_columns.rename(columns=COLUMN_MAPPING_TO_COMMON)

    # Ensure common numeric columns are numeric and present
    common_numeric_cols_expected = ['age', 'weight', 'height', 'bmi']
    for col in common_numeric_cols_expected:
        if col in df_common_columns.columns:
            df_common_columns[col] = pd.to_numeric(df_common_columns[col], errors='coerce')
        else:
            # If a mapped column (like 'bmi' from 'imcINdex') isn't present after renaming,
            # it means the source column was missing or not in COLUMN_MAPPING_TO_COMMON.
            # For AI model, these features are crucial.
            print(f"Warning: Crucial common column '{col}' not found after mapping. AI model might fail.")
            df_common_columns[col] = np.nan # Add as NaN if missing

    # Ensure timestamp is datetime in common df as well
    if 'timestamp' in df_common_columns.columns:
         df_common_columns['timestamp'] = pd.to_datetime(df_common_columns['timestamp'], errors='coerce')


    return df_original_columns, df_common_columns


# The PDF loading function from the new code can be kept if you plan to support PDF uploads later
# For now, we will comment it out to focus on the CSV.
# import pdfplumber
# def load_data_from_pdf_new_structure(pdf_path: str) -> pd.DataFrame:
# rows = []
# with pdfplumber.open(pdf_path) as pdf:
# for page in pdf.pages:
# table = page.extract_table()
# if table:
# for row in table:
# rows.append(row)
# headers = [
# "id_case", "age", "sex", "agreement", "weight",
# "height", "bmi_category", "timestamp", "symptom_1", "symptom_2", "bmi"
# ]
# df = pd.DataFrame(rows, columns=headers)
# if not df.empty and df.iloc[0].equals(pd.Series(headers)):
# df = df.iloc[1:]
# return df

if __name__ == "__main__":
    # Example usage:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(script_dir)
    # Assuming your data folder is at the project root, sibling to src/
    csv_file_path_main = os.path.join(project_root_dir, "data", "doctor31_cazuri(1).csv")

    print(f"Attempting to load data from: {csv_file_path_main}")
    df_orig, df_common = load_and_prepare_data(csv_file_path_main)

    if df_orig is not None:
        print("\n--- Original Columns DataFrame ---")
        print(df_orig.head())
        print(df_orig.info())

    if df_common is not None:
        print("\n--- Common Columns DataFrame ---")
        print(df_common.head())
        print(df_common.info())

        # Test with validators
        from validators import validate_data
        # Ensure validators work with the columns it expects
        df_validated_common = validate_data(df_common.copy())
        print("\n--- Validated Data (common columns) Sample ---")
        expected_cols = ['age', 'weight', 'height', 'bmi', 'valid', 'suspect_elderly_obese']
        actual_cols = [col for col in expected_cols if col in df_validated_common.columns]
        print(df_validated_common[actual_cols].head())