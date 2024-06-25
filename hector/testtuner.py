import tensorflow as tf
from tensorflow import keras
from keras_tuner import BayesianOptimization

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
import numpy as np

import yaml
import argparse
from mlsurfacelayer.mvco_data_QC import load_derived_data_random_test_train_val
from sklearn.ensemble import RandomForestRegressor
from mlsurfacelayer.models import DenseNeuralNetwork
from sklearn.preprocessing import StandardScaler



parser = argparse.ArgumentParser()
parser.add_argument("config", help="Config yaml file")
args = parser.parse_args()
with open(args.config, "r") as config_file:
    config = yaml.load(config_file,Loader=yaml.FullLoader)

# Load dataset
data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                    devVar=config['k_fold_cross_validation']['devVar'],
                                                    N=config['k_fold_cross_validation']['N'],
                                                    k=config['k_fold_cross_validation']['k'],
                                                    scramble=False)

# Define the model builder function
def build_model(hp):
    model_class = model_classes[config["model_type"]]
    model_config = config["model_config"]
    
    # Update model_config with hyperparameters
    if config["model_type"] == "random_forest":
        model_config.update({
            "n_estimators": hp.Int('n_estimators', min_value=10, max_value=100, step=10),
            "max_depth": hp.Int('max_depth', min_value=3, max_value=10, step=1)
        })
    elif config["model_type"] == "neural_network":
        model_config.update({
            "hidden_units": hp.Int('hidden_units', min_value=32, max_value=512, step=32),
            "learning_rate": hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        })

# Define the hyperparameter tuning function
def hyperparameter_tuning(data, config):
    tuner = BayesianOptimization(
        build_model,
        objective='val_loss',  # Adjust based on your task (e.g., 'val_accuracy' for classification)
        max_trials=10,
        directory='my_dir',
        project_name='hyperparam_tuning'
    )
    
    tuner.search(data["train"], data["validate"])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_hps

# Assuming config is defined and contains necessary keys
best_hyperparameters = hyperparameter_tuning(data, config)

# Print best hyperparameters
print(f"Best Hyperparameters: {best_hyperparameters}")

# Use best hyperparameters to build the final model
model = build_model(best_hyperparameters)
model.fit(data["train"])

# Evaluate on test data
test_loss = model.evaluate(data["test"])
print(f"Test Loss: {test_loss}")