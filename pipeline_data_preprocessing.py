import numpy as np
import pandas as pd

from loguru import logger
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest

from datetime import datetime


class DataPreprocessingPipeline():
    def __init__(self, file_path, log_file_path=None):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.log_file_path = log_file_path

        self.preprocessing_info = {
            'original_shape': self.data.shape,
            'missing_values_before': self.data.isnull().sum().sum(),
            'column_types': {},
            'transformations': [],
            'removed_features': [],
            'feature_importances': {}
        }

        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []

        self.initial_preprocessing()

    def initial_preprocessing(self):
        df = self.data.copy()

        id_cols = [col for col in df.columns if 'id' in col.lower()
                   and df[col].nunique() == df.shape[0]]
        if id_cols:
            df = df.drop(columns=id_cols)
            self.preprocessing_info['removed_features'].extend(id_cols)
            self.preprocessing_info['transformations'].append(
                f"Dropped likely ID columns: {id_cols}")

        self.numeric_features = df.select_dtypes(
            include=['float', 'int']).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col])
                self.datetime_features.append(col)
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
            except:
                continue

        self.preprocessing_info['column_types'] = {
            'numeric': self.numeric_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features
        }

        for col in self.datetime_features:
            df[col] = pd.to_datetime(df[col])
            for comp in ['year', 'month', 'day', 'dayofweek']:
                df[f"{col}_{comp}"] = getattr(df[col].dt, comp)
                self.numeric_features.append(f"{col}_{comp}")
            df = df.drop(columns=[col])
            self.preprocessing_info['transformations'].append(
                f"Processed datetime column: {col}")

        self.data = df

    def missing_data_summary(self):
        df = self.data.copy()
        total = df.isnull().sum()
        percent = (total / df.shape[0]) * 100
        table = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        table['Types'] = [str(df[col].dtype) for col in df.columns]
        return table

    def imputation_strategy(self, strategy='mean'):
        df = self.data.copy()

        if isinstance(strategy, dict):
            for col, strat in strategy.items():
                if strat == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                else:
                    imputer = SimpleImputer(
                        strategy=strat, fill_value=0 if strat == 'constant' else None)
                df[[col]] = imputer.fit_transform(df[[col]])
        else:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(
                    strategy=strategy, fill_value=0 if strategy == 'constant' else None)
            df[self.numeric_features] = imputer.fit_transform(
                df[self.numeric_features])

        self.preprocessing_info['transformations'].append(
            f"Imputed missing values using {strategy}")
        self.data = df

    def handle_outlier(self, method):
        df = self.data.copy()

        if method == 'iqr':
            for feature in self.numeric_features:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[feature] = np.where(df[feature] < lower, lower, df[feature])
                df[feature] = np.where(df[feature] > upper, upper, df[feature])
        elif method == 'zscore':
            for feature in self.numeric_features:
                z = stats.zscore(df[feature])
                df[feature] = np.where(
                    z > 3, df[feature].mean() + 3*df[feature].std(), df[feature])
                df[feature] = np.where(
                    z < -3, df[feature].mean() - 3*df[feature].std(), df[feature])
        elif method == 'isolation_forest':
            iso = IsolationForest(contamination=0.05, random_state=42)
            outliers = iso.fit_predict(df[self.numeric_features])
            for feature in self.numeric_features:
                median = df[feature].median()
                df.loc[outliers == -1, feature] = median

        self.preprocessing_info['transformations'].append(
            f"Handled outliers using {method}")
        self.data = df

    def handle_scaling(self, method):
        df = self.data.copy()

        scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(),
                  'robust': RobustScaler()}[method]
        df[self.numeric_features] = scaler.fit_transform(
            df[self.numeric_features])
        self.preprocessing_info['transformations'].append(
            f"Scaled features using {method}")
        self.data = df

    def categorical_features(self, encoding, target_column=None):
        df = self.data.copy()

        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[self.categorical_features] = cat_imputer.fit_transform(
            df[self.categorical_features])
        if encoding == 'one_hot':
            df = pd.get_dummies(
                df, columns=self.categorical_features, drop_first=True)
        elif encoding == 'label':
            for col in self.categorical_features:
                df[col] = df[col].astype('category').cat.codes
        elif encoding == 'binary':
            for col in self.categorical_features:
                df[col] = df[col].astype('category').cat.codes.astype(
                    'int').astype('str').apply(lambda x: list(format(int(x), 'b')))
        elif encoding == 'target' and target_column:
            for col in self.categorical_features:
                means = df.groupby(col)[target_column].mean()
                df[col] = df[col].map(means)

        self.preprocessing_info['transformations'].append(
            f"Encoded categorical features using {encoding}")
        self.data = df

    def feature_selection(self, variance_threshold=0.01, correlation_threshold=0.95):
        df = self.data.copy()

        # Keep only numeric columns for selection
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Variance Threshold
        selector = VarianceThreshold(threshold=variance_threshold)
        selected = selector.fit_transform(numeric_df)
        keep = selector.get_support()
        selected_cols = numeric_df.columns[keep]
        removed = numeric_df.columns[~keep].tolist()

        # Update df with selected numeric + all non-numeric
        df = df[selected_cols.tolist(
        ) + [col for col in df.columns if col not in numeric_df.columns]]

        self.preprocessing_info['removed_features'].extend(removed)

        if correlation_threshold < 1:
            corr_matrix = df[selected_cols].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(
                upper[column] > correlation_threshold)]
            df = df.drop(columns=to_drop)
            self.preprocessing_info['removed_features'].extend(to_drop)

        self.preprocessing_info['transformations'].append(
            "Performed feature selection")
        self.data = df

    def get_report(self):
        self.preprocessing_info['final_shape'] = self.data.shape
        self.preprocessing_info['missing_values_after'] = self.data.isnull(
        ).sum().sum()
        self.preprocessing_info['report_timestamp'] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        return self.preprocessing_info
