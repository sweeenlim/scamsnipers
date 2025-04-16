# scamsnipers

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
├── notebooks                 # Jupyter notebooks with naming
│
├── references.txt            # Data dictionaries, manuals, and other explanatory materials
│
├── reports                   # Generated analysis reports as HTML, PDF, LaTeX, etc.
│   ├── report.md (MEIJIN)    # Markdown to display visualizations
│   └── figures               # Figures and graphics for reporting
│
├── requirements.txt          # Python dependencies for project
│
├── setup.py                  # Makes project installable with `pip install -e .`
│
├── src                       # Source code
│   ├── __init__.py           # Makes `src` a Python module
│   ├── data                  # Scripts to download or generate data
│   │   ├── make_dataset.py (FREEDY)
│   │   ├── clean.py (MARCUS)
│   ├── features              # Scripts to build features from raw data
│   │   ├── build_features.py (FREEDY)
│   ├── models                # Scripts to train and predict with models
│   │   └── models.py (YUCAI, NATALIE, MARCUS)
│   └── visualization         # Scripts to generate visualizations
│       └── visualize.py (SWEEEN)
│
└── tox.ini                   # Configuration file for running `tox` tests
```
