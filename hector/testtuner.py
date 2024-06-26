"""
This file needs to receive the 1 outer yaml. 
    Will use the config to run a hyperparameter tuning search, 
    and inside the tuner will load all kfolds and return the average performance.
    This is how the randomly selected parameters will be evaluated to decide the best.
Normally this yaml is used to create 10 yamls that have the kfold validation in there, 
    I wrote it this way so that it would fit with DJ's pipeline in train_..._model.py
"""
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

import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlsurfacelayer.metrics import mean_error, hellinger_distance, pearson_r2

metrics = {"mean_squared_error": mean_squared_error,
           "mean_absolute_error": mean_absolute_error,
           "pearson_r2": pearson_r2,
           "hellinger_distance": hellinger_distance,
           "mean_error": mean_error}


parser = argparse.ArgumentParser()
parser.add_argument("config", help="Config yaml file")
parser.add_argument("model_type", default=None, help="Model type to be tuned")
parser.add_argument("predictand_type", default=None, help="Variable we want to predict in this tuner")
args = parser.parse_args()
with open(args.config, "r") as config_file:
    config = yaml.load(config_file,Loader=yaml.FullLoader)

if args.model_type == None:
    print('No model type was given. Search will be cancelled.')
    sys.exit()

elif args.predictand_type == None:
    print('No predictand type was given. Search will be cancelled.')
    sys.exit()

# Defind model classes
model_classes = {
    "random_forest": RandomForestRegressor,
    "neural_network": DenseNeuralNetwork
}

# Define the model builder function
def build_model(hp, config):
    model_class = model_classes[args.model_type]
    model_config = config["model_config"][model_class].copy()
    
    # Update model_config with hyperparameters
    if model_class == "random_forest":
        pass
        model_config.update({
            "n_estimators": hp.Int('n_estimators', min_value=10, max_value=100, step=10),
            "max_depth": hp.Int('max_depth', min_value=3, max_value=10, step=1)
        })

        model = model_classes[model_class](**model_config)


    elif model_class == "neural_network":
        model_config.update({
            "hidden_units": hp.Int('hidden_units', min_value=32, max_value=512, step=32),
            "learning_rate": hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        })

        model = model_classes[model_class](**model_config) # build the model

        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
        loss = hp.Choice('loss', values=['mean_squared_error', 'mean_absolute_error'])
        
        if optimizer == 'adam':
            optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer_instance = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)

        model.compile(optimizer=optimizer_instance, loss=loss) # compile the model with selected params

    return model


# Define the cross-validation function
def cross_val_score(hp, config):
    
    scores = [] # empty list to hold each kfold's performance

    model = build_model(hp, config) # select random hyperparameters

    output_type = args.predictand_type
    model_class = model_classes[args.model_type]

    input_column = config["input_columns"][output_type]
    output_column = config["output_columns"][output_type]

    metric = 'mean_absolute_error' # temp until a metric is selected, MAE for testing
    
    for i in range(config['k_fold_cross_validation']['N']):
        # Load dataset
        data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                        devVar=config['k_fold_cross_validation']['devVar'],
                                                        N=config['k_fold_cross_validation']['N'],
                                                        k=i,
                                                        scramble=config['k_fold_cross_validation']['scramble'])
        
        # model_configs = config["model_config"]
        # model_objects = dict()
        # model_predictions.loc[:, output_column] = data["test"][output_column] # trying to save target, but in tuner we dont care
        if model_class == "random_forest":
            model.fit(data["train"][input_column].values,
                      data["train"][output_column].values)
            
            pred = model.predict(data["test"][input_column])
            score = metrics[metric](data["test"][output_column].values, pred.values)

            # score = model.score(X_val, y_val)

        elif model_class == "neural_network":
            input_scaler = StandardScaler()
            var_scale_list = input_column + [output_column]
            scaled_train  = input_scaler.fit_transform( data["train"][var_scale_list])
            scaled_val  = input_scaler.transform( data["validate"][var_scale_list])
            scaled_test = input_scaler.transform(data["test"][var_scale_list])
            
            history = model.fit(scaled_train[:,0:-1], scaled_train[:,-1], scaled_val[:,0:-1], scaled_val[:,-1])
            # Run neural network on scaled test data, fill in the model_predictions column with output
            pred = model.predict(scaled_test[:,0:-1])
            pred = pred  * input_scaler.scale_[len(input_scaler.scale_) - 1 ] + input_scaler.mean_[len(input_scaler.mean_) - 1 ]
            
            # score = model.evaluate(X_val, y_val, verbose=0)
            # scaled_test = pd.DataFrame()
            # for model_metric in model_metric_types:
            #     scaled_test[predictandLabel_model, model_metric] = metrics[model_metric](data["test"][output_columns[output_type]].values, model_predictions[predictandLabel_model].values)
            # for model_metric in model_metric_types:
            # scaled_test.loc[:, (predictandLabel_model, model_metric)] = metrics[model_metric](data["test"][output_columns[output_type]].values, pred.values)
            # predictandLabel_model = output_type + "-" + model_class
            score = metrics[metric](data["test"][output_column].values, pred.values)
                
            # scaled_test.to_csv(join(out_dir, predictandLabel_model + "_scaled_metrics_NN.csv"), index=False)
        
        scores.append(score)
    
    return np.mean(scores)


# Define the hyperparameter tuning function
def hyperparameter_tuning(config):
    def build_model_and_evaluate(hp):
        return cross_val_score(hp, config)

    tuner = BayesianOptimization(
        build_model_and_evaluate,
        objective='val_loss',  # Adjust based on your task (e.g., 'val_accuracy' for classification)
        max_trials=10,
        executions_per_trial=3,  # Evaluate each set of hyperparameters 3 times
        directory='my_dir',
        project_name='hyperparam_tuning'
    )

    # Placeholder data required by the search function, it won't be used
    dummy_data = np.zeros((1, 1))
    tuner.search(dummy_data, dummy_data)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_hps


# Assuming config is defined and contains necessary keys
best_hyperparameters = hyperparameter_tuning(config)

# Print best hyperparameters
print(f"Best Hyperparameters: {best_hyperparameters}")


# the cobe block below is not relevant for us since we will use the best 
#   parameters to run an experiment on casper that does cross validation again
"""
# Load the dataset again for the final training
data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                               devVar=config['k_fold_cross_validation']['devVar'],
                                               N=config['k_fold_cross_validation']['N'],
                                               k=config['k_fold_cross_validation']['k'],
                                               scramble=False)

# Use best hyperparameters to build the final model
final_model = build_model(best_hyperparameters)
final_model.fit(data["train"]["features"], data["train"]["labels"])

# Evaluate on test data
test_loss = final_model.evaluate(data["test"]["features"], data["test"]["labels"])
print(f"Test Loss: {test_loss}")
"""