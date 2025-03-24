import os
import sys
import json
import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_config

# Load the configuration
config = load_config('config.yaml')
logging.info(f"Loaded config: {config}")

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.train_data_path = config['data_transformation']['train_array_file_path']
        self.test_data_path = config['data_transformation']['test_array_file_path']
        self.param_file_path = config['model_trainer']['param_file_path']
        self.evaluation_file_path = config['model_evaluation']['evaluation_file_path']

    def load_params(self):
        """Load hyperparameters from the param.yaml file."""
        try:
            with open(self.param_file_path, 'r') as file:
                params = yaml.safe_load(file)
            return params['models']
        except Exception as e:
            logging.error(f"Error while loading parameters from {self.param_file_path}")
            raise CustomException(e, sys)

    def evaluate_model(self):
        try:
            # Load parameters
            model_params = self.load_params()

            # Load the train and test data
            logging.info(f"Loading train data from: {self.train_data_path}")
            train_arr = np.load(self.train_data_path)
            logging.info(f"Loading test data from: {self.test_data_path}")
            test_arr = np.load(self.test_data_path)

            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Define classifiers
            model_classes = {
                "LogisticRegression": LogisticRegression,
                "DecisionTreeClassifier": DecisionTreeClassifier,
                "RandomForestClassifier": RandomForestClassifier,
                "GradientBoostingClassifier": GradientBoostingClassifier,
                "AdaBoostClassifier": AdaBoostClassifier,
                "ExtraTreesClassifier": ExtraTreesClassifier,
                "XGBClassifier": XGBClassifier,
                "LGBMClassifier": LGBMClassifier,
                "CatBoostClassifier": CatBoostClassifier,
                
            }

            # Dictionary to store metrics for each model
            all_metrics = {}

            # Iterate over each model and its parameters
            for i in range(len(list(model_classes))):
                model_class = list(model_classes.values())[i]  # Get the model class
                para = model_params[list(model_classes.keys())[i]]  # Get corresponding parameters

                # Create an instance of the model class
                model = model_class()

                # Perform grid search
                gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                logging.info(f"Grid search completed for model {model_class.__name__}. Best parameters: {gs.best_params_}")

                # Set the best parameters from grid search
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                logging.info(f"Model {model_class.__name__} trained successfully")

                # Make predictions
                y_test_pred = model.predict(X_test)

                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, average='weighted')
                recall = recall_score(y_test, y_test_pred, average='weighted')
                f1 = f1_score(y_test, y_test_pred, average='weighted')

                # Save the metrics for the current model
                metrics = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                }

                # Add the metrics for the current model to the all_metrics dictionary
                model_name = model_class.__name__
                all_metrics[model_name] = metrics

                logging.info(f"Metrics for model {model_class.__name__}: {metrics}")

                logging.info(f"Model {model_class.__name__} evaluated successfully")


                logging.info(f"Logging parameters and metrics with MLflow for model {model_class.__name__}")

                # Log parameters and metrics with MLflow
                with mlflow.start_run():
                    mlflow.log_param("model_name", model_class.__name__)  # Log the model class name
                    for param, value in gs.best_params_.items():
                        mlflow.log_param(param, value)  # Log each individual parameter
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                logging.info(f"Model {model_class.__name__} evaluated. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

                logging.info(f"mlflow run completed for model {model_class.__name__}")
            
            # Save all metrics for all models to a JSON file
            os.makedirs(os.path.dirname(self.evaluation_file_path), exist_ok=True)
            output_path = self.evaluation_file_path
            with open(output_path, 'w') as f:
                json.dump(all_metrics, f, indent=4)

            logging.info(f"Model evaluation completed. All results saved to {output_path}")

        except Exception as e:
            logging.error(f"Error occurred during model evaluation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        model_evaluation = ModelEvaluation(config)
        model_evaluation.evaluate_model()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")