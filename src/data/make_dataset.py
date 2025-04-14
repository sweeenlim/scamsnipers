import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the root
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

# Set Kaggle credentials in environment
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

# Define dataset path
dataset_path = ROOT_DIR / 'data' / 'raw'
dataset_path.mkdir(parents=True, exist_ok=True)

dataset_slug = os.getenv('KAGGLE_DATASET')  
if not dataset_slug:
    raise ValueError("KAGGLE_DATASET is not set in .env")

# Download and unzip to dataset_path
command = f'kaggle datasets download -d {dataset_slug} --unzip -p "{dataset_path}"'
os.system(command)