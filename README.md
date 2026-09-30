# Hidden Poor Classification

This project explores and compares machine-learning models for identifying households classified as poor using the 2023 Philippine Family Income and Expenditure Survey (FIES). It includes the processed data, trained models, evaluation results, and figures.

## Download the project

### Clone with Git

Install [Git](https://git-scm.com/downloads), open a terminal, and run:

```powershell
git clone https://github.com/jairo3224/hidden_poor.git
cd hidden_poor
```

### Download as a ZIP

Open the [GitHub repository](https://github.com/jairo3224/hidden_poor), select **Code**, then **Download ZIP**. Extract the downloaded ZIP file to a folder on your computer.

## Data

The original FIES household-summary CSV is not included in this repository because it exceeds GitHub's per-file size limit. To run the EDA or preprocessing scripts, obtain the dataset from the Philippine Statistics Authority (PSA) and save it as:

```text
data/FIES PUF 2023 Volume2 Household Summary.CSV
```

The processed feature and label files used for training, along with the test split, are included in `data/`. Therefore, training, evaluation, and SHAP analysis can use those files without the original raw CSV.

## Run the analysis

Use Python with the packages imported by the scripts: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `catboost`, `xgboost`, `joblib`, and `shap`. Install them with:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost joblib shap
```

The scripts use paths relative to the `notebooks` directory. Run them from that directory:

```powershell
cd notebooks
python 01_eda.py
python 02_preprocessing.py
python 03_training.py
python 04_evaluation.py
python 05_shap.py
```

Scripts `01_eda.py` and `02_preprocessing.py` require the raw CSV described above. The later scripts use the processed files in `data/`. Running the analysis may replace generated files in `data/` and `outputs/`.

## Repository contents

- `notebooks/` — scripts for exploratory analysis, preprocessing, model training, evaluation, and SHAP explainability.
- `data/` — processed feature, label, and test-set CSV files.
- `outputs/models/` — saved Logistic Regression, CatBoost, and XGBoost models and the scaler.
- `outputs/results/` — model comparison metrics and hidden-poor records.
- `outputs/figures/` — generated plots and analysis figures.
