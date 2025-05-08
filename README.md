# 🧹 Data Preprocessing Pipeline

A modular, flexible, and extensible Python class for preprocessing structured tabular data. This pipeline is designed to handle common preprocessing steps including:

- Missing value imputation
- Categorical encoding
- Outlier handling
- Feature scaling
- Feature selection
- Datetime feature engineering
- Exploratory reporting

## 📦 Features

- **Auto detection of feature types**: Identifies numeric, categorical, and datetime columns.
- **Customizable missing data imputation**: Supports mean, median, constant, and KNN imputation.
- **Outlier handling**: Options include IQR clipping, Z-score smoothing, and Isolation Forest.
- **Encoding for categorical features**: One-hot, label, binary, and target encoding supported.
- **Feature selection**: Variance threshold and correlation filtering to remove redundant or uninformative features.
- **Scalers**: Standard, Min-Max, and Robust scaling options.
- **Logging & reporting**: Tracks all transformations, removed features, and changes in data shape.

## 🧪 Example Usage

A synthetic dataset is generated and processed using the pipeline:

```python
from pipeline_data_preprocessing import DataPreprocessingPipeline

dpp = DataPreprocessingPipeline("synthetic_customer_data.csv")

# View missing data summary
dpp.missing_data_summary()

# Apply preprocessing steps
dpp.imputation_strategy()               # Impute missing values
dpp.handle_outlier('iqr')              # Handle outliers with IQR method
dpp.handle_scaling('standard')         # Standardize numeric features
dpp.feature_selection()                # Perform feature selection

# View final processed data
df_cleaned = dpp.data

# Get processing report
report = dpp.get_report()
print(report)
```

## 🏗️ File Structure

```
├── pipeline_data_preprocessing.py     # Core class implementation
├── data_preprocess.ipynb              # Jupyter Notebook to demonstrate usage
├── synthetic_customer_data.csv        # Sample dataset (generated in notebook)
└── README.md                          # This file
```

## 🧠 Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- loguru

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn scipy loguru
```

## 📁 Sample Dataset

The notebook generates a synthetic customer dataset including:

- Demographics (age, income, gender)
- Membership levels
- Transaction values
- Churn label

It includes missing values and outliers for testing.

## 📊 Example Output

Example report returned by `get_report()`:

```json
{
  "original_shape": [200, 8],
  "missing_values_before": 60,
  "column_types": {
    "numeric": [...],
    "categorical": [...],
    "datetime": ["signup_date"]
  },
  "transformations": [
    "Dropped likely ID columns: ['customer_id']",
    "Processed datetime column: signup_date",
    "Imputed missing values using mean",
    "Handled outliers using iqr",
    "Scaled features using standard",
    "Performed feature selection"
  ],
  "removed_features": ["customer_id", ...],
  "feature_importances": {},
  "final_shape": [200, XX],
  "missing_values_after": 0,
  "report_timestamp": "2025-05-08 15:12:00"
}
```

## 📜 License

MIT License. Feel free to use and extend this pipeline for your own projects.
