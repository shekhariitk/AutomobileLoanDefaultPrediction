# Data preprocessing
import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from src.utils.main_utils import logging
from src.utils.main_utils import CustomException
from src.utils.main_utils import load_config

paths  = load_config('config.yaml')


class DataPreprocessor:
    def __init__(self, config_path, output_dir='artifacts/data_preprocessing/'):
        """
        Initializes the DataPreprocessor with a configuration file and output directory.

        :param config_path: Path to the YAML schema file.
        :param output_dir: Directory where the preprocessed data will be saved.
        """
        self.config_path = config_path
        self.output_dir = output_dir

        try:
            self.config = self._load_config(config_path)
            logging.info("Configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise CustomException(f"Error loading configuration: {e}")

    def _load_config(self, config_path):
        """
        Load the YAML configuration file.

        :param config_path: Path to the schema configuration file.
        :return: Parsed YAML content.
        """
        import yaml
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            raise CustomException(f"Failed to load config file {config_path}: {e}")

    def _handle_null_values(self, df):
        """Handle missing values by imputation (or you could drop them based on strategy)."""

        # Imputer for numerical columns (using median strategy)
        imputer = SimpleImputer(strategy='median')  # Or 'median'/'most_frequent' as per needs
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

        # Imputer for categorical columns (using most_frequent strategy)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


        logging.info("Null values handled.")
        return df

    def _handle_duplicates(self, df):
        """Remove duplicate rows."""
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        final_rows = df.shape[0]
        logging.info(f"Removed {initial_rows - final_rows} duplicate rows.")
        return df

    def _handle_outliers(self, df):
        """Handle outliers using IQR method (capping or removal)."""
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[:, column] = np.clip(df[column], lower_bound, upper_bound)  # Capping outliers

        logging.info("Outliers handled.")
        return df

    def _handle_data_types(self, df):
        """Ensure that the data types match the schema."""
        for column, expected_type in self.config["columns"].items():
            if column in df.columns:
                df.loc[:, column] = df[column].astype(expected_type)
        logging.info("Data types corrected.")
        return df


    def _handle_imbalance(self, df):
        """Handle class imbalance using SMOTENC to keep categorical variables intact."""
        target_column = self.config["target_column"]

        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Identify categorical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_indices = [X.columns.get_loc(col) for col in categorical_cols]  # Get index positions

            # Label encode the target column if it's categorical
            le = None
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)

                # Save the encoding mapping
                label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                logging.info(f"Label Encoding Mapping: {label_mapping}")
                print(f"Label Encoding Mapping: {label_mapping}")  # Print for visibility

            # Apply SMOTENC (handling categorical features correctly)
            smote_nc = SMOTENC(categorical_features=cat_indices, sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote_nc.fit_resample(X, y)

            # Convert back to DataFrame
            df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            df_balanced[target_column] = y_resampled  # Keep target column numeric

            # Reverse label encoding if target was originally categorical
            if le is not None:
                df_balanced[target_column] = le.inverse_transform(y_resampled)
                df = df_balanced

            logging.info("Class imbalance handled by SMOTENC.")
            return df

        else:
            logging.warning(f"Target column '{target_column}' not found in DataFrame.")
            return df

    def preprocess(self, train_df, test_df):
        """
        Perform all preprocessing steps and save the preprocessed data.

        :param train_df: Training DataFrame to preprocess.
        :param test_df: Test DataFrame to preprocess.
        :return: Paths to the preprocessed train and test data CSV files.
        """
        try:
            # Apply preprocessing steps to both train and test data
            logging.info("Starting preprocessing...")

            # Handle train data preprocessing
            train_df = self._handle_null_values(train_df)
            train_df = self._handle_duplicates(train_df)
            train_df = self._handle_outliers(train_df)
            train_df = self._handle_data_types(train_df)
            #train_df = self._handle_imbalance(train_df)

            # Handle test data preprocessing (same steps, but no imbalance treatment for test set)
            test_df = self._handle_null_values(test_df)
            test_df = self._handle_duplicates(test_df)
            test_df = self._handle_outliers(test_df)
            test_df = self._handle_data_types(test_df)

            # Save preprocessed data
            preprocessed_train_path = f"{self.output_dir}/preprocessed_train.csv"
            preprocessed_test_path = f"{self.output_dir}/preprocessed_test.csv"

            os.makedirs(os.path.dirname(preprocessed_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(preprocessed_test_path), exist_ok=True)

            train_df.to_csv(preprocessed_train_path, index=False)
            test_df.to_csv(preprocessed_test_path, index=False)

            logging.info(f"Preprocessing completed. Preprocessed data saved to:\n{preprocessed_train_path}\n{preprocessed_test_path}")

            return preprocessed_train_path, preprocessed_test_path

        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise CustomException(f"Error during preprocessing: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the preprocessor with the path to the schema/config file
        config_path = 'config/schema.yaml'
        preprocessor = DataPreprocessor(config_path)

        # Example: Preprocess training and test data
        train_df = pd.read_csv(paths['data_ingestion']['train_data_path'])
        test_df = pd.read_csv(paths['data_ingestion']['test_data_path'])

        # Preprocess data and get the file paths for the preprocessed datasets
        preprocessed_train_path, preprocessed_test_path = preprocessor.preprocess(train_df, test_df)

        logging.info(f"Preprocessed train data saved to: {preprocessed_train_path}")
        logging.info(f"Preprocessed test data saved to: {preprocessed_test_path }")

        print(f"Preprocessed train data saved to: {preprocessed_train_path}")
        print(f"Preprocessed test data saved to: {preprocessed_test_path}")
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
