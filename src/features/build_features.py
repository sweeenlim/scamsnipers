import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

# Load root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Define the path to the raw data directory
RAW_DATA_DIR = ROOT_DIR / 'data' / 'interim'


def transform_policy_bind_date(df):
    df["incident_minus_policy_bind_date"] = (pd.to_datetime(df["incident_date"])-pd.to_datetime(df["policy_bind_date"])).dt.days
    return df

# Segment the age into different ranges
def segment_age(df):
    bins = list(range(15,66,5))
    labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['age_range'] = df['age_range'].astype(str)
    df['age_range'] = df['age'].apply(lambda age: 'young adults' if 15 <= age <= 25 else 'middle aged adults' if 26 <= age <= 45 else 'older adults')
    return df

# Segment the months as customer into different ranges
def segment_months_as_customer(df):
    bins = list(range(0, 481,60))

    labels = ['0-60', '61-120', '121-180', '181-240', '241-300', '301-360', '361-420', '421-480']

    df['months_as_customer_range'] = pd.cut(df['months_as_customer'], bins=bins, labels=labels, right=False)
    df['months_as_customer_range'] = df['months_as_customer_range'].astype(str)
    df['months_as_customer_range'] = pd.Categorical(df['months_as_customer_range'], categories=labels, ordered=True)
    df['months_as_customer_range'] = df['months_as_customer'].apply(lambda months: 'short term' if 0 <= months <= 60 else 'mid term' if 61 <= months <= 300 else 'long term')
    return df

def transform_hobby(df):
    df['insured_hobbies'] = df['insured_hobbies'].apply(lambda x: x if x == 'chess' or x == 'cross-fit' else 'others')
    return df

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
    df = transform_policy_bind_date(df) # transform policy bind date and incident date diff
    df = segment_age(df) # segment age into ranges
    df = segment_months_as_customer(df) # segment months as customer into ranges
    df = transform_hobby(df) # transform hobbies into 3 categories
    df = df.drop(['policy_number','policy_bind_date', 'incident_date','incident_location'], axis = 1)  
    df = df.drop(['age', 'months_as_customer'], axis = 1) # drop the original columns
    df = encode_incident_sev(df) # encode incident severity
    df = combine_auto_cols(df) # combine auto columns
    df = one_hot_encode_categorical_cols(df) # one-hot encode categorical columns

    # Transform the target variable 'fraud_reported' to binary
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
    print('Starting feature engineering...')
    # Load the raw data
    df = load_raw_data('cleaned_data.csv')
    print(df)
    # Clean the data
    cleaned_df = feature_engineering(df)
    # Save the processed data
    save_fe_data(cleaned_df)
    print('Feature engineering completed.')
    

if __name__ == "__main__":
    main()

