import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

# Load root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Define the path to the raw data directory
RAW_DATA_DIR = ROOT_DIR / 'data' / 'interim'

# Load data from the raw data directory
def load_raw_data(filename):
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File {filename} does not exist in {RAW_DATA_DIR}")
    df = pd.read_csv(file_path)
    return df

# To encode the incident severity in following ordinal format
# 'Trivial Damage' = 0, 'Minor Damage' = 1, 'Major Damage' = 2, 'Total Loss' = 3
def encode_incident_sev(df):
    encoder = OrdinalEncoder(categories=[['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']])
    df['incident_severity_encoded'] = encoder.fit_transform(df[['incident_severity']])
    df = df.drop(['incident_severity'], axis = 1)
    return df

# To combine the auto columns into a single column as they are highly correlated
def combine_auto_cols(df):
    df['auto_make_model'] = df['auto_make'] + '_' + df['auto_model']
    df = df.drop(['auto_make', 'auto_model'], axis = 1)
    return df

# To one-hot encode the categorical columns
def one_hot_encode_categorical_cols(df):
    categorical_cols = df.select_dtypes(include='object').columns
    one_hot_col = [col for col in categorical_cols if col != 'fraud_reported' and col != 'incident_severity_encoded']
    df_encoded = pd.get_dummies(df, columns=one_hot_col, drop_first=True)
    return df_encoded    


def feature_engineering(df): 
    df = encode_incident_sev(df) # encode incident severity
    df = combine_auto_cols(df) # combine auto columns
    df = one_hot_encode_categorical_cols(df) # one-hot encode categorical columns

    # Trnsform the target variable 'fraud_reported' to binary
    df['fraud_reported'] = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

    return df

def save_fe_data(df):
    # Define the path to the processed data directory
    PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the cleaned data to the processed data directory
    file_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df.to_csv(file_path, index=False)
    print(f"Feature engineered data saved to {file_path}")
    return

def main():
    # Load the raw data
    df = load_raw_data('cleaned_data.csv')
    print(df)
    # Clean the data
    cleaned_df = feature_engineering(df)
    # Save the processed data
    save_fe_data(cleaned_df)
    return

if __name__ == "__main__":
    main()

