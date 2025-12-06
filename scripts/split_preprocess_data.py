# split_preprocess_data.py
# Splits validated data into train/test sets and applies standard scaling to numerical features.
# Creates and saves a fitted preprocessing pipeline for consistent data transformation.
# Saves raw and scaled datasets along with the fitted preprocessor.

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import pickle
import os

@click.command()
@click.option('--validated-data', type=str, help="Path to validated data CSV file")
@click.option('--data-to', type=str, help="Path to directory where train and test data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--test-size', type=float, default=0.3, help="Proportion of data to use for test set (default: 0.3)")
@click.option('--random-state', type=int, default=123, help="Random seed for reproducibility (default: 123)")

def main(validated_data, data_to, preprocessor_to, test_size, random_state):
    """
    Split validated data into train/test sets, create, fit a preprocessor 
    and save outputs.
    
    Performs train-test split, creates a StandardScaler preprocessing 
    pipeline for numerical features, fits the preprocessor on training data, 
    transforms both train and test sets, and saves all outputs including raw splits, 
    scaled data and the fitted preprocessor.
    
    Parameters
    ----------
    validated_data : str
        Path to the validated CSV file containing maternal health data.
    data_to : str
        Directory path where train/test splits (both raw and scaled) will be saved.
        The directory will be created if it does not exist.
    preprocessor_to : str
        Directory path where the fitted preprocessor pickle file will be saved.
        The directory will be created if it does not exist.
    test_size : float
        Proportion of the dataset to include in the test split.
        It has a default value of 0.3
    random_state : int
        Random seed for reproducible train-test splitting.
        It has a default value of 123.
    
    Returns
    -------
    None
        The function saves 4 CSV files (train, test, scaled_train, scaled_test) 
        to `data_to` directory and one pickle file (preprocessor) to 
        `preprocessor_to` directory.
    """

    # create output directories if they do not already exist
    os.makedirs(data_to, exist_ok=True)
    os.makedirs(preprocessor_to, exist_ok=True)

    # read the validated data
    data = pd.read_csv(validated_data)

    # split the data
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)

    # save train and test data
    train_path = os.path.join(data_to, "maternal_health_risk_train.csv")
    test_path = os.path.join(data_to, "maternal_health_risk_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # separate features and target
    X_train = train_df.drop(columns=['RiskLevel'])
    y_train = train_df['RiskLevel']
    X_test = test_df.drop(columns='RiskLevel')
    y_test = test_df['RiskLevel']

    # create the preprocessor
    preprocessor = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include='number')),
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # save the preprocessor
    preprocessor.fit(X_train)
    preprocessor_path = os.path.join(preprocessor_to, "maternal_risk_preprocessor.pickle")
    pickle.dump(preprocessor, open(preprocessor_path, "wb"))

    # transform the data
    scaled_X_train = preprocessor.transform(X_train)
    scaled_X_test = preprocessor.transform(X_test)

    # convert back to DataFrames with proper column names
    scaled_train_df = pd.DataFrame(scaled_X_train, columns=preprocessor.get_feature_names_out())
    scaled_test_df = pd.DataFrame(scaled_X_test, columns=preprocessor.get_feature_names_out())

    # add target column back
    scaled_train_df['RiskLevel'] = y_train.values
    scaled_test_df['RiskLevel'] = y_test.values

    # save the transformed data
    scaled_train_path = os.path.join(data_to, "scaled_maternal_health_risk_train.csv")
    scaled_test_path = os.path.join(data_to, "scaled_maternal_health_risk_test.csv")
    
    scaled_train_df.to_csv(scaled_train_path, index=False)
    scaled_test_df.to_csv(scaled_test_path, index=False)

if __name__ == '__main__':
    main()