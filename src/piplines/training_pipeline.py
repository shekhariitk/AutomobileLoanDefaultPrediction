from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_trainer import ModelTrainerClass
from src.components.data_validation import DataValidator
from src.components.data_transformation import DataTransformation
import sys
from src.logger import logging
from src.utils.main_utils import load_config
from src.exception import CustomException
import pandas as pd
import numpy as np


paths = load_config('config.yaml')


class TrainingPipeline:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def run_training_pipeline(self):
        # Data Ingestion
        try:
            logging.info("Initiating Data Ingestion...")
            data_ingestion = DataIngestion(self.config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed successfully. Train data path: {train_data_path}, Test data path: {test_data_path}")
        except Exception as e:
            logging.error(f"Error during Data Ingestion: {e}")
            raise CustomException(f"Error during Data Ingestion: {e}")
        
        # Data Validation
        try:
            validator = DataValidator(self.config)
            logging.info("Initiating Data Validation...")

            # Example: Validate training data
            train_df = pd.read_csv(self.config['data_ingestion']['train_data_path'])
            train_results = validator.validate(train_df, data_type='train')

            # Example: Validate testing data
            test_df = pd.read_csv(self.config['data_ingestion']['test_data_path'])
            test_results = validator.validate(test_df, data_type='test')

            logging.info("Data Validation completed successfully.")
        
        except Exception as e:
            print(f"Error in validation process: {e}")

        # Data Preprocessing
        try:
        
            preprocessor = DataPreprocessor(self.config)

            # Example: Preprocess training and test data
            train_df = pd.read_csv(self.config['data_ingestion']['train_data_path'])
            test_df = pd.read_csv(self.config['data_ingestion']['test_data_path'])

            # Preprocess data and get the file paths for the preprocessed datasets
            preprocessed_train_path, preprocessed_test_path = preprocessor.preprocess(train_df, test_df)
        
        except Exception as e:
            print(f"Error during preprocessing: {e}")

        # Data Transformation
        try:
            data_transformation = DataTransformation(self.config)
            train_path = self.config['data_preprocessing']['preprocessed_train_data_path']
            test_path = self.config['data_preprocessing']['preprocessed_test_data_path']

            train_arr, test_arr, preprocessor_ob_path = data_transformation.initiate_data_transformation(train_path, test_path)

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise CustomException(e, sys) from e
        
        try:
            model_trainer = ModelTrainerClass()
            train_arr = paths['data_transformation']['train_array_file_path']
            test_arr = paths['data_transformation']['test_array_file_path']
            train_arr = np.load(train_arr)
            test_arr = np.load(test_arr)
            model_trainer.initiate_model_training(train_arr, test_arr)

        except Exception as e:
           logging.error(f"An error occurred: {str(e)}")
           raise CustomException(e, sys) from e
        
        logging.info("Training pipeline completed successfully.")
        print("Training pipeline completed successfully.")
