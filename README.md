# scamsnipers

## Overview 
This project provides a comprehensive solution for detecting insurance fraud claims ([Kaggle](https://www.kaggle.com/code/buntyshah/insurance-fraud-claims-detection)) using machine learning techniques. The approach covers the data science pipeline, from initial data analysis and exploration to feature engineering and model development. The focus is to build a robust fraud detection system, to reduce operational costs and enhance decision-making processes within insurance companies.

## Project Structure

```plaintext
├── LICENSE                   # License for the project
├── Makefile                  # Makefile for managing essential workflows
├── README.md                 # Project documentation
│
├── data                      # Data storage folder
│   ├── interim               # Intermediate data after data cleaning
│   ├── processed             # Final data for modeling after feature engineering
│   └── raw                   # The original, immutable data dump
│
├── .env.template             # Template for .env file
│
├── docs                      # Sphinx documentation
│
├── notebook                  # Jupyter notebook for entire pipeline
│
├── references.txt            # Data dictionaries, manuals, and other explanatory materials
│
├── reports                   # Generated analysis reports as HTML, PDF, LaTeX, etc.
│   ├── report.md             # Markdown to display visualizations
│   └── figures               # Figures and graphics for reporting
│
├── requirements.txt          # Python dependencies for project
│
├── setup.py                  # Makes project installable with `pip install -e .`
│
├── src                       # Source code
│   ├── data                  # Scripts to download or generate data
│   │   ├── make_dataset.py 
│   │   ├── clean.py 
│   │   ├── datadictionary.txt 
│   ├── features              # Scripts to build features from raw data
│   │   ├── build_features.py 
│   ├── models                # Scripts to train and predict with models
│   │   └── models.py 
│   └── visualization         # Scripts to generate visualizations
│       └── visualize.py
│
└── tox.ini                   # Configuration file for running `tox` tests
```
## Set up
To include when MAKEFILE is done