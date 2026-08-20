"""Create the cleaned, outlier-capped dataset used by the dashboard."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
RAW_DATA_PATH = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"
OUTPUT_DATA_PATH = DATA_DIR / "ObesityDataSet_Cleaned_Outliers_Capped.csv"


def cap_outliers_iqr(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap a column at its 1.5 IQR lower and upper bounds."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    capped_data = data.copy()
    capped_data[column] = capped_data[column].clip(
        lower=lower_bound, upper=upper_bound
    )
    return capped_data


def prepare_dataset() -> pd.DataFrame:
    # Data cleaning: load raw data and remove duplicate records.
    cleaned_data = pd.read_csv(RAW_DATA_PATH)
    initial_rows = len(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates().copy()

    # Combination step: create BMI before applying the outlier treatment.
    combined_data = cleaned_data.copy()
    combined_data["BMI"] = combined_data["Weight"] / combined_data["Height"] ** 2

    # Cap Age, Height, Weight, and BMI with the 1.5 IQR rule.
    outliers_handled = combined_data.copy()
    for column in ["Age", "Height", "Weight", "BMI"]:
        outliers_handled = cap_outliers_iqr(outliers_handled, column)

    outliers_handled.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Initial raw data rows: {initial_rows}")
    print(f"Rows after dropping duplicates: {len(cleaned_data)}")
    print(f"Saved cleaned dataset: {OUTPUT_DATA_PATH}")
    return outliers_handled


if __name__ == "__main__":
    prepare_dataset()
