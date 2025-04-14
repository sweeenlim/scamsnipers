# scamsnipers

## Project Structure

```plaintext
├── LICENSE                   # License for the project
├── Makefile                  # Makefile for managing essential workflows
├── README.md                 # Project documentation
│
├── data                      # Data storage folder
│   ├── external              # Data from third-party sources
│   ├── interim               # Intermediate data that has been transformed
│   ├── processed             # The final, canonical data sets for modeling
│   └── raw                   # The original, immutable data dump
│
├── docs                      # Sphinx documentation
│   
├── notebooks                 # Jupyter notebooks with naming
│
├── references.txt            # Data dictionaries, manuals, and other explanatory materials
│
├── reports                   # Generated analysis reports as HTML, PDF, LaTeX, etc.
│   └── figures               # Figures and graphics for reporting
│
├── requirements.txt          # Python dependencies for project
│
├── setup.py                  # Makes project installable with `pip install -e .`
│
├── src                       # Source code
│   ├── __init__.py           # Makes `src` a Python module
│   ├── data                  # Scripts to download or generate data
│   │   └── make_dataset.py   #
│   ├── features              # Scripts to build features from raw data
│   │   └── build_features.py 
│   ├── models                # Scripts to train and predict with models
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization         # Scripts to generate visualizations
│       └── visualize.py
│
└── tox.ini                   # Configuration file for running `tox` tests
```