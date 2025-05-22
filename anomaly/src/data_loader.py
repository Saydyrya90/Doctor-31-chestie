import pdfplumber
import pandas as pd


def load_data_from_pdf(pdf_path: str) -> pd.DataFrame:
    rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    rows.append(row)

    # FORȚĂM headerele corecte (scrise manual)
    headers = [
        "id_case", "age", "sex", "agreement", "weight",
        "height", "bmi_category", "timestamp", "symptom_1", "symptom_2", "bmi"
    ]

    df = pd.DataFrame(rows, columns=headers)

    # Eliminăm primul rând dacă e duplicat (același cu headerele)
    if df.iloc[0].equals(pd.Series(headers)):
        df = df.iloc[1:]

    return df
