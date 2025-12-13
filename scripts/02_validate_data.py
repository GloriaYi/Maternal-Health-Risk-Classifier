# validate_data.py
# Validates raw maternal health data for correct format and data quality.
# Checks for outliers, missing values, duplicates and class imbalance.
# Saves cleaned data and logs validation errors for review.

import click
import pandas as pd
import pandera.pandas as pa
import logging
import json
import os

def validate_file_format(file_path, expected_file_extension):
    """
    Function to check if the file has the correct format,
    given the file path and expected file extension.
    """
    if not file_path.endswith(expected_file_extension):
        error_msg = "Invalid file format."
        raise ValueError(error_msg)

def validate_column_names(data, expected_columns):
    """
    Function to check if the DataFrame has the correct column names,
    given the DataFrame and the expected column names.
    """
    actual_columns = data.columns.tolist()
    
    missing_columns = set(expected_columns) - set(actual_columns)
    if missing_columns:
        error_msg = "Missing required columns."
        raise ValueError(error_msg)

    extra_columns = set(actual_columns) - set(expected_columns)
    if extra_columns:
        error_msg = "Unexpected extra columns found."
        raise ValueError(error_msg)

@click.command()
@click.option('--raw-data', type=str, help="Path to raw data CSV file")
@click.option('--data-to', type=str, help="Path to directory where validated data will be written to")
@click.option('--log-to', type=str, help="Path to directory where validation log will be written to")

def main(raw_data, data_to, log_to):
    """
    Validate raw maternal health data and save validated output.

    Performs comprehensive data validation including file format checks,
    column name validation, schema validation for data types and ranges,
    duplicate detection, missingness checks, and class imbalance detection.
    Invalid rows are filtered out and validation errors are logged.

    Parameters
    ----------
    raw_data : str
        Path to the raw CSV file containing maternal health data.
    data_to : str
        Path to directory where the validated data CSV will be saved.
        The directory will be created if it does not exist.
    log_to : str
        Path to directory where the validation error logs will be saved.
        The directory will be created if it does not exist.

    Returns
    -------
    None
        The function saves validated data to `validated_data.csv` in the
        `data_to` directory and logs and validated errors to 
        `validation_errors.log` in the `log_to` directory.

    Notes
    -----
    Expected columns: Age, SystolicBP, DiastolicBP, BS, BodyTemp, 
                      HeartRate, RiskLevel
    
    Validation checks include:
    - Correct file format (.csv)
    - Expected column names
    - Valid ranges for numerical features
    - Valid categories for RiskLevel (low risk, mid risk, high risk)
    - No more than 5% missing values per column
    - No duplicate or empty rows
    - Minimum 5% representation for each risk level category
    - No constant-valued features
    """

    # create output directories if they do not already exist
    os.makedirs(data_to, exist_ok=True)
    os.makedirs(log_to, exist_ok=True)

    # configure logging
    log_file = os.path.join(log_to, "validation_errors.log")
    logging.basicConfig(
        filename=log_file, 
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    # read the data
    health_data = pd.read_csv(raw_data, header=0)

    # expected file extension and columns
    expected_file_extension = ".csv"
    expected_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

    # define the schema
    schema = pa.DataFrameSchema(
        {
            # check for correct category labels
            "RiskLevel": pa.Column(str, pa.Check.isin(["low risk", "mid risk", "high risk"]), nullable=True),
            # check for outliers and anomalous values
            "Age": pa.Column(int, pa.Check.between(10, 65), nullable=True),
            "SystolicBP": pa.Column(int, pa.Check.between(60, 200), nullable=True),
            "DiastolicBP": pa.Column(int, pa.Check.between(40, 140), nullable=True),
            "BS": pa.Column(float, pa.Check.between(1.0, 25.0), nullable=True),
            "BodyTemp": pa.Column(float, pa.Check.between(95.0, 105.0), nullable=True),
            "HeartRate": pa.Column(int, pa.Check.between(50, 150), nullable=True)
        },
        checks=[
            # check for duplicate rows and empty rows
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
            # check for missingness not beyond expected threshold
            pa.Check(lambda df: (df.isna().mean() <= 0.05).all(), error="Some columns have more than 5% missing values."),
            # check for label imbalance in target variable
            pa.Check(
                lambda df: (df["RiskLevel"].value_counts(normalize=True) >= 0.05).all(),
                error="One or more RiskLevel categories have <5% of observations.",
            ),
    
            # make sure there is no column with constant values
            pa.Check(
                lambda df: df[["Age", "SystolicBP", "DiastolicBP", "BS",
                              "BodyTemp", "HeartRate"]].nunique().min() > 1,
                error="One or more features have no variation.",
            ),
        ],
        drop_invalid_rows=False,
    )

    # initialize error cases DataFrame
    error_cases = pd.DataFrame()
    data = health_data.copy()
    
    # validate data and handle errors: correct file format, correct column names
    try:
        validate_file_format(raw_data, expected_file_extension)
        validate_column_names(data, expected_columns)
    except ValueError as e:
        # convert the error message to a JSON string
        error_message = json.dumps(str(e), indent=2)
        logging.error("\n" + error_message)
    
    # validate data and handle errors: no empty observations, missingness not beyond threshold,
    # correct data types for columns, no outliers, correct category levels
    try:
        validated_data = schema.validate(data, lazy=True)
    except pa.errors.SchemaErrors as e:
        error_cases = e.failure_cases
    
        # convert the error message to a JSON string
        error_message = json.dumps(e.message, indent=2)
        logging.error("\n" + error_message)
    
    # filter out invalid rows based on the error cases: keep the duplicate rows 
    if not error_cases.empty:
        invalid_indices = error_cases["index"].dropna().unique()
        validated_data = (
            data.drop(index=invalid_indices)
            .reset_index(drop=True)
            .dropna(how="all")
        )
    else:
        validated_data = data

    # save validated data
    output_file = os.path.join(data_to, "validated_data.csv")
    validated_data.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()