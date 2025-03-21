# Data validation

import pandas as pd
import numpy as np
import json
import yaml
from src.logger import logging
import os
from src.exception import CustomException
from src.utils.main_utils import load_config

paths  = load_config('config.yaml')

class DataValidator:
    def __init__(self, config_path, output_dir='artifacts/data_validation/'):
        """
        Initializes the DataValidator with a configuration file and output directory.

        :param config_path: Path to the YAML schema file.
        :param output_dir: Directory where validation results will be saved.
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
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            raise

    def _check_null_values(self, df):
        """Check for null values in the dataframe."""
        try:
            null_values = df.isnull().sum().to_dict()
            return null_values
        except Exception as e:
            logging.error(f"Error checking null values: {e}")
            raise CustomException(f"Error checking null values: {e}")

    def _check_duplicates(self, df):
        """Check for duplicated rows in the dataframe."""
        try:
            duplicated_rows = df[df.duplicated()].shape[0]
            return duplicated_rows
        except Exception as e:
            logging.error(f"Error checking duplicated rows: {e}")
            raise CustomException(f"Error checking duplicated rows: {e}")

    def _check_outliers(self, df):
        """Check for outliers in numerical columns using the IQR method."""
        try:
            outliers = {}
            for column in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
                outliers[column] = outlier_count
            return outliers
        except Exception as e:
            logging.error(f"Error checking outliers: {e}")
            raise CustomException(f"Error checking outliers: {e}")

    def _check_data_types(self, df):
        """Check if the data types match the schema in the config file."""
        try:
            column_types = df.dtypes.to_dict()
            data_type_validation = {}
            for column, expected_type in self.config["columns"].items():
                if column in column_types:
                    actual_type = str(column_types[column])
                    if actual_type != expected_type:
                        data_type_validation[column] = {"expected": expected_type, "actual": actual_type}
                else:
                    data_type_validation[column] = {"status": "column not found"}
            return data_type_validation
        except Exception as e:
            logging.error(f"Error checking data types: {e}")
            raise CustomException(f"Error checking data types: {e}")

    def _check_imbalance(self, df):
        """Check for imbalance in the target column."""
        try:
            target_column = self.config["target_column"]
            if target_column in df.columns:
                target_counts = df[target_column].value_counts()
                imbalance_ratio = target_counts.min() / target_counts.max()
                return imbalance_ratio
            else:
                return {"status": "target column not found"}
        except Exception as e:
            logging.error(f"Error checking imbalance in target column: {e}")
            raise CustomException(f"Error checking imbalance in target column: {e}")

    def validate(self, df, data_type='train'):
        """
        Perform the complete data validation process.

        :param df: DataFrame to validate.
        :param data_type: Type of data (train or test), used for naming the output file.
        :return: Dictionary containing validation results.
        """
        validation_results = {}

        try:
            validation_results["null_values"] = self._check_null_values(df)
            validation_results["duplicated_rows"] = self._check_duplicates(df)
            validation_results["outliers"] = self._check_outliers(df)
            validation_results["data_type_validation"] = self._check_data_types(df)
            validation_results["target_column_imbalance"] = self._check_imbalance(df)

            # Save the validation results as a JSON file
            output_file = f"{self.output_dir}/{data_type}_validation.json"

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'w') as json_file:
                json.dump(validation_results, json_file, indent=4)

            logging.info(f"Validation for {data_type} data completed successfully.")
            return validation_results

        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            raise CustomException(f"Error during data validation: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the validator with the path to the schema/config file
        config_path = 'config/schema.yaml'
        validator = DataValidator(config_path)

        # Example: Validate training data
        train_df = pd.read_csv(paths['data_ingestion']['train_data_path'])
        train_results = validator.validate(train_df, data_type='train')

        # Example: Validate testing data
        test_df = pd.read_csv(paths['data_ingestion']['test_data_path'])
        test_results = validator.validate(test_df, data_type='test')
    
    except Exception as e:
        print(f"Error in validation process: {e}")