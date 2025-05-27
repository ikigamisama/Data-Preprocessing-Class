# DataPreprocessor

A robust, easy-to-use data preprocessing pipeline for machine learning projects. This library provides a comprehensive solution for cleaning, transforming, and preparing your data with minimal code and maximum flexibility.

## ✨ Key Features

- **🔗 Method Chaining**: Fluent interface for readable preprocessing pipelines
- **🤖 Smart Automation**: Automatic feature type detection and intelligent defaults
- **🎯 Flexible Target Handling**: Support for single, multiple, or all columns as targets
- **🛡️ Robust Error Handling**: Graceful handling of edge cases and data issues
- **📊 Comprehensive Logging**: Track every transformation with detailed logs
- **⚡ Memory Efficient**: Optimized operations that don't bloat memory usage
- **🔧 Highly Configurable**: Fine-tune every aspect of preprocessing

## 🚀 Quick Start

```python
from data_preprocessor import DataPreprocessor

# Load and preprocess data in one line
preprocessor = DataPreprocessor('your_data.csv', target_column='target')
processed_data = preprocessor.quick_preprocess()

# Or build a custom pipeline
processed_data = (DataPreprocessor('your_data.csv', target_column='target')
                  .clean_data()
                  .handle_missing_values('auto')
                  .handle_outliers('iqr')
                  .encode_categorical('auto')
                  .scale_features('standard')
                  .get_processed_data())
```

## 📦 Installation

```bash
# Required dependencies
pip install pandas numpy scikit-learn scipy
```

## 🎯 Target Column Options

The `target_column` parameter supports multiple formats:

```python
# Single target column
DataPreprocessor(data, target_column='price')

# Multiple target columns
DataPreprocessor(data, target_column=['price', 'quantity'])

# All columns (no exclusions)
DataPreprocessor(data, target_column='all')

# No target (default)
DataPreprocessor(data)
```

## 🔧 Core Methods

### Data Cleaning

```python
.clean_data(
    remove_duplicates=True,    # Remove duplicate rows
    remove_empty_rows=True,    # Remove completely empty rows
    remove_id_columns=True     # Remove auto-detected ID columns
)
```

### Missing Value Handling

```python
.handle_missing_values(
    strategy='auto',           # 'auto', 'mean', 'median', 'mode', 'drop', 'knn'
    threshold=0.5             # Drop columns with >50% missing values
)

# Custom strategies per column
.handle_missing_values({
    'age': 'median',
    'income': 'mean',
    'category': 'most_frequent'
})
```

### Outlier Detection & Treatment

```python
.handle_outliers(
    method='iqr',             # 'iqr', 'zscore', 'isolation_forest'
    contamination=0.1         # For isolation forest
)
```

### Categorical Encoding

```python
.encode_categorical(
    method='auto',            # 'auto', 'onehot', 'label', 'target'
    max_categories=10         # Max categories for one-hot encoding
)
```

### Feature Scaling

```python
.scale_features(
    method='standard',        # 'standard', 'minmax', 'robust'
    feature_list=None         # Specific features to scale
)
```

### Feature Selection

```python
.remove_low_variance(
    threshold=0.01           # Remove features with variance < threshold
)
```

## 📋 Complete Examples

### Example 1: Quick Preprocessing

```python
import pandas as pd
from data_preprocessor import DataPreprocessor

# Load your data
preprocessor = DataPreprocessor('sales_data.csv', target_column='revenue')

# Apply standard preprocessing pipeline
processed_data = preprocessor.quick_preprocess()

print(f"Original shape: {preprocessor.original_shape}")
print(f"Processed shape: {processed_data.shape}")
```

### Example 2: Custom Pipeline

```python
# Build a custom preprocessing pipeline
preprocessor = DataPreprocessor(df, target_column='target')

processed_data = (preprocessor
    .clean_data(remove_id_columns=True)
    .handle_missing_values({
        'numerical_col': 'median',
        'categorical_col': 'most_frequent'
    })
    .handle_outliers('isolation_forest', contamination=0.05)
    .encode_categorical('onehot', max_categories=5)
    .scale_features('robust')
    .remove_low_variance(0.01)
    .get_processed_data())
```

### Example 3: Process All Columns

```python
# When you want to preprocess all columns uniformly
preprocessor = DataPreprocessor(df, target_column='all')
processed_data = preprocessor.quick_preprocess()
```

### Example 4: Exploratory Data Analysis

```python
# Get detailed information about your preprocessing
preprocessor = DataPreprocessor('data.csv')
preprocessor.quick_preprocess()

# View processing summary
summary = preprocessor.get_summary()
print(f"Processing steps: {summary['processing_steps']}")
print(f"Features removed: {summary['final_columns'] - summary['original_columns']}")

# View detailed log
for step in summary['processing_log']:
    print(step)
```

## 🧠 Smart Features

### Automatic Feature Type Detection

The preprocessor automatically identifies:

- **Numeric features**: int, float columns
- **Categorical features**: object, category columns with reasonable cardinality
- **Datetime features**: Auto-detected date strings (extracts year, month, day, dayofweek)
- **ID columns**: High-cardinality columns likely to be identifiers

### Intelligent Defaults

- **Missing values**: Median for numeric, mode for categorical
- **Categorical encoding**: One-hot for low cardinality, label encoding for high cardinality
- **Outlier handling**: IQR method (caps outliers instead of removing data points)
- **Scaling**: Standard scaling for numeric features

### Robust Processing

- Handles edge cases gracefully (empty datasets, all-missing columns, etc.)
- Memory-efficient operations
- Preserves data integrity throughout the pipeline
- Comprehensive error messages and warnings

## 📊 Data Types Supported

| Data Type       | Handling               | Notes                                   |
| --------------- | ---------------------- | --------------------------------------- |
| **Numeric**     | Standard preprocessing | int64, float64                          |
| **Categorical** | Encoding + imputation  | object, category                        |
| **Datetime**    | Feature extraction     | Converts to year, month, day, dayofweek |
| **Boolean**     | Treated as categorical | bool                                    |
| **ID columns**  | Auto-removal           | High cardinality columns                |

## 🎛️ Configuration Options

### Missing Value Strategies

- `'auto'`: Median for numeric, mode for categorical
- `'mean'`: Mean imputation
- `'median'`: Median imputation
- `'mode'` / `'most_frequent'`: Mode imputation
- `'drop'`: Remove rows with missing values
- `'knn'`: KNN imputation (k=5)

### Outlier Methods

- `'iqr'`: Interquartile range (caps at Q1-1.5*IQR and Q3+1.5*IQR)
- `'zscore'`: Z-score method (caps at ±3 standard deviations)
- `'isolation_forest'`: Isolation Forest algorithm

### Encoding Methods

- `'auto'`: Smart choice based on cardinality
- `'onehot'`: One-hot encoding
- `'label'`: Label encoding
- `'target'`: Target encoding (requires target column)

### Scaling Methods

- `'standard'`: StandardScaler (mean=0, std=1)
- `'minmax'`: MinMaxScaler (scale to 0-1)
- `'robust'`: RobustScaler (uses median and IQR)

## 🔍 Monitoring & Debugging

### Comprehensive Logging

Every preprocessing step is logged with timestamps:

```python
preprocessor = DataPreprocessor(data, verbose=True)
# Outputs:
# [14:30:15] Data loaded from DataFrame
# [14:30:15] Dataset shape: (1000, 8)
# [14:30:15] Numeric features: 3
# [14:30:15] Categorical features: 2
# [14:30:16] Removed 15 duplicate rows
# [14:30:16] Applied median imputation to numeric features
# ...
```

### Processing Summary

```python
summary = preprocessor.get_summary()
print(summary)
# {
#     'original_shape': (1000, 8),
#     'final_shape': (985, 12),
#     'processing_steps': 8,
#     'numeric_features': 9,
#     'categorical_features': 0,
#     'missing_values': 0,
#     'processing_log': [...]
# }
```

## ⚠️ Best Practices

### 1. Handle Missing Values First

```python
# Good: Handle missing values before other operations
preprocessor.handle_missing_values('auto').handle_outliers('iqr')

# Avoid: Outlier detection on data with missing values
```

### 2. Scale After Encoding

```python
# Good: Encode categoricals first, then scale
preprocessor.encode_categorical('auto').scale_features('standard')
```

### 3. Check Data Quality

```python
# Always examine your data first
print(f"Missing values:\n{preprocessor.data.isnull().sum()}")
print(f"Data types:\n{preprocessor.data.dtypes}")
```

### 4. Use Method Chaining

```python
# Clean and readable
processed_data = (preprocessor
    .clean_data()
    .handle_missing_values('auto')
    .encode_categorical('auto')
    .scale_features('standard')
    .get_processed_data())
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Target column 'X' not found"

```python
# Solution: Check column names
print(preprocessor.data.columns.tolist())
```

**Issue**: Memory errors with large datasets

```python
# Solution: Process in chunks or use more memory-efficient methods
preprocessor.handle_missing_values('median')  # Instead of 'knn'
```

**Issue**: All features removed after preprocessing

```python
# Solution: Check variance threshold
preprocessor.remove_low_variance(threshold=0.001)  # Lower threshold
```

## 🤝 Contributing

Feel free to contribute by:

- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning preprocessing tools
- **numpy**: Numerical computing
- **scipy**: Scientific computing

---

**Happy Preprocessing! 🎉**

_Make your data ready for machine learning with just a few lines of code._
