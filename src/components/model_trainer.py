import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import save_obj, evaluate_model,load_config
from dataclasses import dataclass
from jsonschema import validate, ValidationError

# Load the configuration
config = load_config('config.yaml')
logging.info(f"Loaded config: {config}")

# Model Training Configuration
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(config['model_trainer']['trained_model_file_path'])
    param_file_path = os.path.join(config['model_trainer']['param_file_path'])
    schema_file_path = os.path.join(config['model_trainer']['schema_file_path'])

# Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def load_params(self):
        """Load hyperparameters from the param.yaml file."""
        try:
            with open(self.model_trainer_config.param_file_path, 'r') as file:
                params = yaml.safe_load(file)
            return params['models']
        except Exception as e:
            logging.error(f"Error while loading parameters from {self.model_trainer_config.param_file_path}")
            raise CustomException(e, sys)

    def validate_params(self, params):
        """Validate parameters against schema.yaml."""
        try:
            with open(self.model_trainer_config.schema_file_path, 'r') as schema_file:
                schema = yaml.safe_load(schema_file)

            validate(instance=params, schema=schema['models'])
            logging.info("Parameter validation successful.")
        except ValidationError as ve:
            logging.error(f"Parameter validation failed: {ve.message}")
            raise CustomException(f"Parameter validation failed: {ve.message}", sys)
        except Exception as e:
            logging.error(f"Error during schema validation: {str(e)}")
            raise CustomException(e, sys)

    def select_best_model_based_on_metrics(self, model_report: dict, models: dict):
        """
        Selects the best model based on recall (primary), F1 score (secondary), and precision (tertiary).
        Suitable for high-stakes classification tasks like automobile loan defaulter prediction.
        """
        best_model_name = None
        best_recall = -1
        best_f1 = -1
        best_precision = -1
        best_model = None

        for model_name, metrics in model_report.items():
            recall = metrics.get('Recall', 0)
            f1 = metrics.get('F1 Score', 0)
            precision = metrics.get('Precision', 0)

            # Prioritize recall
            if recall > best_recall:
                best_recall, best_f1, best_precision = recall, f1, precision
                best_model_name, best_model = model_name, models[model_name]
            elif recall == best_recall:
                # If recall is tied, prioritize F1 score
                if f1 > best_f1:
                    best_f1, best_precision = f1, precision
                    best_model_name, best_model = model_name, models[model_name]
                elif f1 == best_f1:
                    # If F1 is tied, prioritize precision
                    if precision > best_precision:
                        best_precision = precision
                        best_model_name, best_model = model_name, models[model_name]

        logging.info(f"Best model selected: {best_model_name} | Recall: {best_recall}, F1 Score: {best_f1}, Precision: {best_precision}")
        return best_model

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and dependent variables.")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "ExtraTreesClassifier": ExtraTreesClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "XGBClassifier": XGBClassifier(n_estimators=45, max_depth=5, learning_rate=0.5),
                "LGBMClassifier": LGBMClassifier(),
                "CatBoostClassifier": CatBoostClassifier(silent=True)
            }

            # Load and validate parameters
            params = self.load_params()
            self.validate_params(params)

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models=models, param_grids=params)
            logging.info(f"Model report: {model_report}")

            # Select the best model using recall-based prioritization
            best_model = self.select_best_model_based_on_metrics(model_report, models)
            logging.info(f"Best model: {best_model}")

            # Save the best model
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        model_trainer = ModelTrainerClass()
        train_arr = config['data_transformation']['train_array_file_path']
        test_arr = config['data_transformation']['test_array_file_path']
        train_arr = np.load(train_arr)
        test_arr = np.load(test_arr)
        model_trainer.initiate_model_training(train_arr, test_arr)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise CustomException(e, sys)