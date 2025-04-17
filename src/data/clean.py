import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the root
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

# Define the path to the raw data directory
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'

# Load data from the raw data directory
def load_raw_data(filename):
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File {filename} does not exist in {RAW_DATA_DIR}")
    df = pd.read_csv(file_path)
    return df

# Clean the data
def clean_data(df):
    # remove column _c39 as it contains only null values
    df = df.drop(['_c39'], axis = 1)

    # Replace ? with NaN
    df = df.replace('?', np.nan)
    
    # Check for null values
    print("Null values in the dataset:")
    print(df.isnull().sum())

    # Imputate NaN values
    fill_values = {
        'collision_type': df['collision_type'].mode()[0],  # Mode for collision_type
        'property_damage': 'NO',  # Fill 'NO' for property_damage
        'police_report_available': 'NO',  # Fill 'NO' for police_report_available
        'authorities_contacted': 'Other'  # Fill 'others' for authorities_contacted
    }

    # Apply the fillna operation for all columns in the dictionary
    df.fillna(value=fill_values, inplace=True)
    
    # Verify the changes
    print("Null values in the after cleaning dataset:")
    print(df.isna().sum())

    return df

def save_processed_data(df):
    # Define the path to the processed data directory
    PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'interim'
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the cleaned data to the processed data directory
    file_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")
    return

def main():
    # Load the raw data
    df = load_raw_data('insurance_claims.csv')
    # Clean the data
    cleaned_df = clean_data(df)
    # Save the processed data
    save_processed_data(cleaned_df)
    return

if __name__ == "__main__":
    main()
