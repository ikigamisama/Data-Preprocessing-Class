import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from typing import Union, List, Dict, Optional, Tuple
import warnings
import logging
from datetime import datetime


class DataPreprocessor:
    """
    A robust and easy-to-use data preprocessing pipeline for machine learning.

    Key improvements:
    - Better error handling and validation
    - Cleaner API with sensible defaults
    - Automatic data type detection
    - Comprehensive logging
    - Memory efficient operations
    - Flexible configuration options
    """

    def __init__(self, data: Union[str, pd.DataFrame], target_column: Optional[Union[str, List[str]]] = None,
                 verbose: bool = True):
        """
        Initialize the preprocessor.

        Args:
            data: File path (CSV) or pandas DataFrame
            target_column: Name of target column(s), list of columns, or "all" for all columns
            verbose: Whether to print processing information
        """
        self.verbose = verbose
        self.target_column = self._process_target_column(data if isinstance(
            data, pd.DataFrame) else pd.read_csv(data), target_column)
        self.processing_log = []

        # Load data
        if isinstance(data, str):
            try:
                self.data = pd.read_csv(data)
                self._log(f"Data loaded from {data}")
            except Exception as e:
                raise ValueError(f"Could not load data from {data}: {str(e)}")
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self._log("Data loaded from DataFrame")
        else:
            raise TypeError(
                "Data must be a file path (string) or pandas DataFrame")

        self.original_shape = self.data.shape
        self.original_columns = self.data.columns.tolist()

        # Initialize feature type lists
        self._identify_feature_types()

        self._log(f"Dataset shape: {self.data.shape}")
        self._log(f"Numeric features: {len(self.numeric_features)}")
        self._log(f"Categorical features: {len(self.categorical_features)}")
        self._log(f"Target column(s): {self.target_column}")
        self._log(f"Missing values: {self.data.isnull().sum().sum()}")

    def _process_target_column(self, data: pd.DataFrame, target_column: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Process target column specification."""
        if target_column is None:
            return None
        elif target_column == "all":
            return data.columns.tolist()
        elif isinstance(target_column, str):
            if target_column in data.columns:
                return [target_column]
            else:
                raise ValueError(
                    f"Target column '{target_column}' not found in data")
        elif isinstance(target_column, list):
            missing_cols = [
                col for col in target_column if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Target columns not found: {missing_cols}")
            return target_column
        else:
            raise TypeError(
                "target_column must be string, list of strings, 'all', or None")

    def _log(self, message: str):
        """Internal logging method."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.verbose:
            print(log_entry)

    def _identify_feature_types(self):
        """Automatically identify feature types."""
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.id_features = []

        for col in self.data.columns:
            # Skip target column(s)
            if self.target_column and col in self.target_column:
                continue

            # Check for ID columns (high cardinality, mostly unique)
            if (col.lower().endswith('id') or col.lower().startswith('id') or
                    self.data[col].nunique() / len(self.data) > 0.95):
                self.id_features.append(col)
                continue

            # Check for datetime columns
            if self.data[col].dtype == 'object':
                sample = self.data[col].dropna().head(100)
                try:
                    pd.to_datetime(sample, errors='raise',
                                   infer_datetime_format=True)
                    self.datetime_features.append(col)
                    continue
                except:
                    pass

            # Numeric features
            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Check if it's actually categorical (few unique values)
                unique_ratio = self.data[col].nunique() / len(self.data)
                if unique_ratio < 0.05 and self.data[col].nunique() < 20:
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)

    def clean_data(self, remove_duplicates: bool = True, remove_empty_rows: bool = True,
                   remove_id_columns: bool = True) -> 'DataPreprocessor':
        """
        Basic data cleaning operations.

        Args:
            remove_duplicates: Remove duplicate rows
            remove_empty_rows: Remove rows that are completely empty
            remove_id_columns: Remove identified ID columns
        """
        initial_shape = self.data.shape

        if remove_empty_rows:
            self.data = self.data.dropna(how='all')

        if remove_duplicates:
            before_dups = len(self.data)
            self.data = self.data.drop_duplicates()
            removed_dups = before_dups - len(self.data)
            if removed_dups > 0:
                self._log(f"Removed {removed_dups} duplicate rows")

        if remove_id_columns and self.id_features:
            self.data = self.data.drop(columns=self.id_features)
            self._log(f"Removed ID columns: {self.id_features}")

        if self.data.shape != initial_shape:
            self._log(
                f"Shape changed from {initial_shape} to {self.data.shape}")

        return self

    def handle_missing_values(self, strategy: Union[str, Dict[str, str]] = 'auto',
                              threshold: float = 0.5) -> 'DataPreprocessor':
        """
        Handle missing values with automatic or custom strategies.

        Args:
            strategy: 'auto', 'mean', 'median', 'mode', 'drop', 'knn', or dict mapping columns to strategies
            threshold: Drop columns with missing ratio above this threshold
        """
        # Drop columns with too many missing values
        missing_ratios = self.data.isnull().sum() / len(self.data)
        cols_to_drop = missing_ratios[missing_ratios >
                                      threshold].index.tolist()

        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
            self.numeric_features = [
                f for f in self.numeric_features if f not in cols_to_drop]
            self.categorical_features = [
                f for f in self.categorical_features if f not in cols_to_drop]
            self._log(
                f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")

        # Handle remaining missing values
        if strategy == 'drop':
            initial_len = len(self.data)
            self.data = self.data.dropna()
            self._log(
                f"Dropped {initial_len - len(self.data)} rows with missing values")
            return self

        # Impute missing values
        if isinstance(strategy, str):
            if strategy == 'auto':
                # Auto strategy: median for numeric, mode for categorical
                num_strategy = 'median'
                cat_strategy = 'most_frequent'
            elif strategy == 'knn':
                # Use KNN imputation for all features
                if self.numeric_features:
                    imputer = KNNImputer(n_neighbors=5)
                    self.data[self.numeric_features] = imputer.fit_transform(
                        self.data[self.numeric_features])
                self._log("Applied KNN imputation to numeric features")

                # Mode for categorical
                if self.categorical_features:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    self.data[self.categorical_features] = cat_imputer.fit_transform(
                        self.data[self.categorical_features])
                    self._log("Applied mode imputation to categorical features")
                return self
            else:
                num_strategy = strategy
                cat_strategy = 'most_frequent' if strategy in [
                    'mean', 'median'] else strategy
        else:
            # Custom strategy dictionary
            for col, strat in strategy.items():
                if col in self.data.columns:
                    if strat == 'knn':
                        imputer = KNNImputer(n_neighbors=5)
                        self.data[[col]] = imputer.fit_transform(
                            self.data[[col]])
                    else:
                        imputer = SimpleImputer(strategy=strat)
                        self.data[[col]] = imputer.fit_transform(
                            self.data[[col]])
            self._log(f"Applied custom imputation strategies: {strategy}")
            return self

        # Apply imputation
        if self.numeric_features:
            num_imputer = SimpleImputer(strategy=num_strategy)
            self.data[self.numeric_features] = num_imputer.fit_transform(
                self.data[self.numeric_features])
            self._log(f"Applied {num_strategy} imputation to numeric features")

        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            self.data[self.categorical_features] = cat_imputer.fit_transform(
                self.data[self.categorical_features])
            self._log(
                f"Applied {cat_strategy} imputation to categorical features")

        return self

    def handle_outliers(self, method: str = 'iqr', contamination: float = 0.1) -> 'DataPreprocessor':
        """
        Handle outliers in numeric features.

        Args:
            method: 'iqr', 'zscore', or 'isolation_forest'
            contamination: Proportion of outliers (for isolation forest)
        """
        if not self.numeric_features:
            self._log("No numeric features found for outlier handling")
            return self

        outliers_handled = 0

        if method == 'iqr':
            for col in self.numeric_features:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outlier_mask = (self.data[col] < lower) | (
                    self.data[col] > upper)
                outliers_handled += outlier_mask.sum()

                # Cap outliers instead of removing them
                self.data[col] = np.clip(self.data[col], lower, upper)

        elif method == 'zscore':
            for col in self.numeric_features:
                z_scores = np.abs(stats.zscore(self.data[col]))
                outlier_mask = z_scores > 3
                outliers_handled += outlier_mask.sum()

                # Replace outliers with values at 3 standard deviations
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                self.data.loc[self.data[col] > mean_val +
                              3*std_val, col] = mean_val + 3*std_val
                self.data.loc[self.data[col] < mean_val -
                              3*std_val, col] = mean_val - 3*std_val

        elif method == 'isolation_forest':
            iso_forest = IsolationForest(
                contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(
                self.data[self.numeric_features])

            outlier_mask = outlier_labels == -1
            outliers_handled = outlier_mask.sum()

            # Replace outliers with median values
            for col in self.numeric_features:
                median_val = self.data[col].median()
                self.data.loc[outlier_mask, col] = median_val

        self._log(f"Handled {outliers_handled} outliers using {method}")
        return self

    def encode_categorical(self, method: str = 'auto', max_categories: int = 10) -> 'DataPreprocessor':
        """
        Encode categorical features.

        Args:
            method: 'auto', 'onehot', 'label', or 'target'
            max_categories: Maximum categories for one-hot encoding
        """
        if not self.categorical_features:
            self._log("No categorical features found")
            return self

        # Handle missing values in categorical features first
        for col in self.categorical_features:
            if self.data[col].isnull().any():
                mode_val = self.data[col].mode(
                )[0] if not self.data[col].mode().empty else 'Unknown'
                self.data[col] = self.data[col].fillna(mode_val)

        if method == 'auto':
            # Automatically choose encoding based on cardinality
            for col in self.categorical_features:
                n_unique = self.data[col].nunique()
                if n_unique <= max_categories:
                    # One-hot encode
                    dummies = pd.get_dummies(
                        self.data[col], prefix=col, drop_first=True)
                    self.data = pd.concat(
                        [self.data.drop(columns=[col]), dummies], axis=1)
                else:
                    # Label encode
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])

        elif method == 'onehot':
            for col in self.categorical_features:
                if self.data[col].nunique() <= max_categories:
                    dummies = pd.get_dummies(
                        self.data[col], prefix=col, drop_first=True)
                    self.data = pd.concat(
                        [self.data.drop(columns=[col]), dummies], axis=1)
                else:
                    self._log(
                        f"Skipping one-hot encoding for {col} (too many categories)")

        elif method == 'label':
            for col in self.categorical_features:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])

        elif method == 'target' and self.target_column:
            for col in self.categorical_features:
                # Use first target column for target encoding if multiple targets
                target_col = self.target_column[0] if isinstance(
                    self.target_column, list) else self.target_column
                if target_col in self.data.columns:
                    target_mean = self.data.groupby(col)[target_col].mean()
                    self.data[col] = self.data[col].map(target_mean)
                else:
                    self._log(
                        f"Target column {target_col} not found, skipping target encoding")

        # Update feature lists
        self._identify_feature_types()
        self._log(f"Encoded categorical features using {method}")
        return self

    def scale_features(self, method: str = 'standard', feature_list: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Scale numeric features.

        Args:
            method: 'standard', 'minmax', or 'robust'
            feature_list: Specific features to scale (default: all numeric)
        """
        features_to_scale = feature_list or self.numeric_features

        if not features_to_scale:
            self._log("No numeric features found for scaling")
            return self

        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler = scalers[method]
        self.data[features_to_scale] = scaler.fit_transform(
            self.data[features_to_scale])

        self._log(f"Scaled {len(features_to_scale)} features using {method}")
        return self

    def remove_low_variance(self, threshold: float = 0.01) -> 'DataPreprocessor':
        """Remove features with low variance."""
        if not self.numeric_features:
            return self

        selector = VarianceThreshold(threshold=threshold)
        selected_data = selector.fit_transform(
            self.data[self.numeric_features])

        # Get selected feature names
        selected_features = np.array(self.numeric_features)[
            selector.get_support()].tolist()
        removed_features = [
            f for f in self.numeric_features if f not in selected_features]

        if removed_features:
            self.data = self.data.drop(columns=removed_features)
            self.numeric_features = selected_features
            self._log(f"Removed {len(removed_features)} low-variance features")

        return self

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed dataset."""
        return self.data.copy()

    def get_summary(self) -> Dict:
        """Get a summary of all preprocessing steps."""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.data.shape,
            'original_columns': len(self.original_columns),
            'final_columns': len(self.data.columns),
            'processing_steps': len(self.processing_log),
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features),
            'missing_values': self.data.isnull().sum().sum(),
            'processing_log': self.processing_log
        }

    def quick_preprocess(self, target_column: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Apply a standard preprocessing pipeline with sensible defaults.

        Args:
            target_column: Target column name(s), list of columns, or "all" for all columns
        """
        if target_column is not None:
            self.target_column = self._process_target_column(
                self.data, target_column)

        self._log("Starting quick preprocessing pipeline...")

        (self.clean_data()
         .handle_missing_values('auto')
         .handle_outliers('iqr')
         .encode_categorical('auto')
         .scale_features('standard')
         .remove_low_variance())

        self._log("Quick preprocessing completed!")
        return self.get_processed_data()
