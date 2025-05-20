import pandas as pd
import numpy as np

# --- Constants for Anomaly Detection Rules ---
# Rule R1: Age
MAX_VALID_AGE = 120
MIN_VALID_AGE = 0 # Assuming age can't be 0 or negative for a patient record.
                  # If 0 is valid (e.g. newborns < 1 year old), adjust to < 0.

# Rule R2: Weight
ADULT_AGE_THRESHOLD = 18
MIN_ADULT_WEIGHT_KG = 20  # Based on SRS "5kg for adult" - 20kg is a more robust general minimum.
MAX_ADULT_WEIGHT_KG = 400 # A very high but plausible upper limit.
MIN_CHILD_WEIGHT_KG = 2   # For children (0 < Age < 18).
MIN_WEIGHT_KG = 0         # Absolute minimum, weight cannot be negative.

# Rule R3: Height
MIN_HEIGHT_CM = 50  # SRS mentions < 120cm. 50cm catches more extreme errors.
MAX_HEIGHT_CM = 250 # SRS mentions > 220cm.

# Rule R4: BMI (based on provided 'imcINdex')
MIN_BMI_THRESHOLD = 10 # Client advice was <12, SRS also implies very low is bad.
MAX_BMI_THRESHOLD = 70 # Client advice was >60, SRS also implies very high is bad.

# Rule R5: Elderly Obesity
ELDERLY_AGE_THRESHOLD = 85
SUSPICIOUS_OBESITY_CATEGORIES = ['Obese', 'Extremly Obese'] # Case-sensitive as per data

# Rule R6: Duplicates
DUPLICATE_TIMEFRAME_HOURS = 1

# Rule R7: BMI Calculation Consistency
BMI_CALCULATION_TOLERANCE = 1.0

# --- Helper Function to Add Anomaly Reasons ---
def _add_anomaly_reason(df_row, reason_code, reason_desc):
    """Appends a reason to the 'anomaly_reason' list for a given row if not already present."""
    full_reason = f"{reason_code}: {reason_desc}"
    # Ensure 'anomaly_reason' field is a list
    if not isinstance(df_row['anomaly_reason'], list):
        df_row['anomaly_reason'] = [] # Initialize if it's not a list (e.g. NaN)
    
    if full_reason not in df_row['anomaly_reason']:
        df_row['anomaly_reason'].append(full_reason)
    df_row['is_anomaly'] = True
    return df_row

# --- Rule-Based Anomaly Detection Functions ---

def apply_age_rules(df):
    """Applies age-related anomaly detection rules."""
    # R1.1: Age > MAX_VALID_AGE
    condition = df['age_v'] > MAX_VALID_AGE
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R1.1", f"Age > {MAX_VALID_AGE} years"), axis=1
    )
    # R1.2: Age <= MIN_VALID_AGE (or <0 if 0 is valid for newborns)
    condition = df['age_v'] <= MIN_VALID_AGE
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R1.2", f"Age <= {MIN_VALID_AGE} years"), axis=1
    )
    return df

def apply_weight_rules(df):
    """Applies weight-related anomaly detection rules."""
    condition = df['greutate'] < MIN_WEIGHT_KG
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R2.1", "Negative weight"), axis=1
    )
    condition = (df['age_v'] >= ADULT_AGE_THRESHOLD) & (df['greutate'].notna()) & (df['greutate'] < MIN_ADULT_WEIGHT_KG)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R2.2", f"Adult (Age >= {ADULT_AGE_THRESHOLD}) with Weight < {MIN_ADULT_WEIGHT_KG} kg"), axis=1
    )
    condition = (df['age_v'] >= ADULT_AGE_THRESHOLD) & (df['greutate'].notna()) & (df['greutate'] > MAX_ADULT_WEIGHT_KG)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R2.3", f"Adult (Age >= {ADULT_AGE_THRESHOLD}) with Weight > {MAX_ADULT_WEIGHT_KG} kg"), axis=1
    )
    condition = (df['age_v'] < ADULT_AGE_THRESHOLD) & (df['age_v'] > MIN_VALID_AGE) & \
                (df['greutate'].notna()) & (df['greutate'] < MIN_CHILD_WEIGHT_KG)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R2.4", f"Child ({MIN_VALID_AGE} < Age < {ADULT_AGE_THRESHOLD}) with Weight < {MIN_CHILD_WEIGHT_KG} kg"), axis=1
    )
    return df

def apply_height_rules(df):
    """Applies height-related anomaly detection rules."""
    condition = (df['inaltime'].notna()) & (df['inaltime'] < MIN_HEIGHT_CM)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R3.1", f"Height < {MIN_HEIGHT_CM} cm"), axis=1
    )
    condition = (df['inaltime'].notna()) & (df['inaltime'] > MAX_HEIGHT_CM)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R3.2", f"Height > {MAX_HEIGHT_CM} cm"), axis=1
    )
    return df

def apply_bmi_rules(df):
    """Applies BMI-related anomaly detection rules using 'imcINdex'."""
    if 'imcINdex_cleaned' not in df.columns: # Ensure this column exists
         df['imcINdex_cleaned'] = df['imcINdex'].replace([np.inf, -np.inf], np.nan)

    condition = (df['imcINdex_cleaned'].notna()) & (df['imcINdex_cleaned'] < MIN_BMI_THRESHOLD)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R4.1", f"BMI < {MIN_BMI_THRESHOLD} (using provided imcINdex)"), axis=1
    )
    condition = (df['imcINdex_cleaned'].notna()) & (df['imcINdex_cleaned'] > MAX_BMI_THRESHOLD)
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R4.2", f"BMI > {MAX_BMI_THRESHOLD} (using provided imcINdex)"), axis=1
    )
    return df

def apply_elderly_obesity_rule(df):
    """Applies rule for suspicious elderly obesity."""
    condition = (df['age_v'] > ELDERLY_AGE_THRESHOLD) & \
                (df['IMC'].isin(SUSPICIOUS_OBESITY_CATEGORIES))
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R5.1", f"Age > {ELDERLY_AGE_THRESHOLD} and IMC in {SUSPICIOUS_OBESITY_CATEGORIES}"), axis=1
    )
    return df

def apply_duplicate_case_rule(df):
    """Applies rule for duplicate cases within a short timeframe."""
    df_sorted = df.sort_values(by=['age_v', 'greutate', 'inaltime', 'data1']).copy()
    df_sorted['prev_data1'] = df_sorted.groupby(['age_v', 'greutate', 'inaltime'])['data1'].shift(1)
    df_sorted['time_diff_to_prev_hours'] = (df_sorted['data1'] - df_sorted['prev_data1']).dt.total_seconds() / 3600
    duplicate_indices = df_sorted[
        (df_sorted['prev_data1'].notna()) &
        (df_sorted['time_diff_to_prev_hours'] >= 0) &
        (df_sorted['time_diff_to_prev_hours'] < DUPLICATE_TIMEFRAME_HOURS)
    ].index
    df.loc[df.index.isin(duplicate_indices)] = df.loc[df.index.isin(duplicate_indices)].apply(
        lambda row: _add_anomaly_reason(row, "R6.1", f"Potential duplicate (same age/weight/height within {DUPLICATE_TIMEFRAME_HOURS} hr of another record)"), axis=1
    )
    return df

def apply_bmi_consistency_rule(df):
    """Checks if calculated BMI is consistent with provided 'imcINdex'."""
    if 'inaltime_m' not in df.columns:
        df['inaltime_m'] = df['inaltime'] / 100 # Ensure in meters
    if 'calculated_bmi' not in df.columns:
        df['calculated_bmi'] = np.where(
            (df['inaltime_m'].notna()) & (df['inaltime_m'] > 0) & (df['greutate'].notna()),
            df['greutate'] / (df['inaltime_m'] ** 2),
            np.nan
        )
        df['calculated_bmi'] = df['calculated_bmi'].replace([np.inf, -np.inf], np.nan)
    if 'imcINdex_cleaned' not in df.columns: # Ensure this exists from apply_bmi_rules
        df['imcINdex_cleaned'] = df['imcINdex'].replace([np.inf, -np.inf], np.nan)

    condition = (
        df['calculated_bmi'].notna() &
        df['imcINdex_cleaned'].notna() &
        (df['imcINdex_cleaned'] <= MAX_BMI_THRESHOLD) & # Only if original BMI wasn't already flagged as extreme by R4.2
        (abs(df['calculated_bmi'] - df['imcINdex_cleaned']) > BMI_CALCULATION_TOLERANCE)
    )
    df.loc[condition] = df.loc[condition].apply(
        lambda row: _add_anomaly_reason(row, "R7.1", f"Calculated BMI differs from imcINdex by > {BMI_CALCULATION_TOLERANCE}"), axis=1
    )
    return df