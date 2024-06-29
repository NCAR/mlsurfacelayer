import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
import numpy as np
import yaml
import argparse
import sys
from mlsurfacelayer.mvco_data_QC import load_derived_data_random_test_train_val
from sklearn.ensemble import RandomForestRegressor
from mlsurfacelayer.models import DenseNeuralNetwork
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlsurfacelayer.metrics import mean_error, hellinger_distance, pearson_r2

from tensorflow.keras.layers import Input


class MyHyperModel(kt.HyperModel):
    def __init__(self, config, args, verbose=0):
        self.config = config
        self.verbose = verbose
        self.args = args

    def build(self, hp):
        # Build and return the model instance
        return build_model(hp, self.config, self.args, self.verbose)

    def fit(self, hp, model, *args, **kwargs):
        if self.args.verbose >= 2: print('\n\n\nCalling MyHyperModel.fit() method\n\n')
        scores = []
        for _ in range(self.args.executions_per_trial):
            score = parallel_cross_val_score(hp, self.config, model, self.args, verbose=self.verbose)
            scores.append(score)

        # this code calculates the mean of the 3 executions, but a more typical tuner will take the best
        # should I take the best or the mean for a more representative metric that takes a little of the epistemic uncertainty into account?
        mean_score = np.mean(scores)
        return {'val_loss': mean_score}

# Define the model builder function
def build_model(hp, config, args, verbose=0):
    model_type = args.model_type
    model_config = config["model_config"][model_type]
    
    if model_type == "random_forest":
        model_config.update({
            "n_estimators": hp.Int('n_estimators', min_value=10, max_value=100, step=10),
            "max_depth": hp.Int('max_depth', min_value=3, max_value=10, step=1)
        })
        model = RandomForestRegressor(**model_config)
    
    elif model_type == "neural_network":

        data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                    devVar=config['k_fold_cross_validation']['devVar'],
                                                    N=config['k_fold_cross_validation']['N'],
                                                    k=0,
                                                    scramble=config['k_fold_cross_validation']['scramble'])
        output_type = args.predictand_type
        input_column = config["input_columns"][output_type]
        output_column = config["output_columns"][output_type]
        input_scaler = StandardScaler()
        var_scale_list = input_column + [output_column]
        scaled_train = input_scaler.fit_transform(data["train"][var_scale_list])
        # scaled_val = input_scaler.transform(data["validate"][var_scale_list])
        # scaled_test = input_scaler.transform(data["test"][var_scale_list])

        if args.verbose >= 2: 
            model_config.update({
                "verbose": 2
            })
        model_config.update({
            "hidden_layers": hp.Choice('hidden_layers', values=[1, 3, 5]),
            "lr": hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
            "optimizer": hp.Choice('optimizer', values=['adam', 'sgd']),
            "loss": hp.Choice('loss', values=['mean_squared_error', 'mean_absolute_error'])
        })
        x, y = scaled_train[:, :-1], scaled_train[:, -1]
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        # if self.classifier:
        #     outputs = np.unique(y).size

        model = DenseNeuralNetwork(**model_config).build_neural_network(inputs, outputs)   
    
    if verbose >= 2:
        print(f"Step: Build Model")
        print(f"Model Class: {model_type}")
        print(f"Model Config: {model_config}")
    
    return model

def parallel_cross_val_score(hp, config, model, args, verbose=0):
    if args.verbose >= 2: print('Inside parallel_cross_val_score:')
    scores = []
    for fold in range(config['k_fold_cross_validation']['N']):
        if args.verbose >= 2: print(f'Fold number:{fold} in parallel_cross_val_score')
        score = cross_val_score(hp, config, model, args, fold, verbose=verbose)
        scores.append(score)

        # alternative parallel method, unsure which one is correct
        # for future in as_completed(futures):
        #     scores.append(future.result())
    return np.mean(scores)

# Define the cross-validation function
def cross_val_score(hp, config, model, args, fold, verbose=0):
    # model = build_model(hp, config, verbose=verbose)
    output_type = args.predictand_type
    model_type = args.model_type
    input_column = config["input_columns"][output_type]
    output_column = config["output_columns"][output_type]
    metric = 'mean_absolute_error'
    
    data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                    devVar=config['k_fold_cross_validation']['devVar'],
                                                    N=config['k_fold_cross_validation']['N'],
                                                    k=fold,
                                                    scramble=config['k_fold_cross_validation']['scramble'])

    if model_type == "random_forest":
        model.fit(data["train"][input_column].values, 
                    data["train"][output_column].values)

        pred = model.predict(data["test"][input_column])
        score = metrics[metric](data["test"][output_column].values, pred)

    elif model_type == "neural_network":
        input_scaler = StandardScaler()
        var_scale_list = input_column + [output_column]
        scaled_train = input_scaler.fit_transform(data["train"][var_scale_list])
        scaled_val = input_scaler.transform(data["validate"][var_scale_list])
        scaled_test = input_scaler.transform(data["test"][var_scale_list])
        
        # model.fit(scaled_train[:, :-1], scaled_train[:, 0:-1], # alternative model.fit, not sure which is right
        # model.fit(scaled_train[:, :-1], scaled_train[:, -1], 
        #         validation_data=(scaled_val[:, :-1], scaled_val[:, -1]))
        model.fit(scaled_train[:, :-1], scaled_train[:, -1], scaled_val[:, :-1], scaled_val[:, -1])
        

        # model.fit = DenseNeuralNetwork

        pred = model.predict(scaled_test[:, :-1])
        pred = pred * input_scaler.scale_[-1] + input_scaler.mean_[-1]

        score = metrics[metric](data["test"][output_column], pred)
    
    if verbose >= 2:
        print(f"Step: Cross Validation")
        print(f"Fold: {fold}")
        print(f"Model Class: {model_type}")
        print(f"Input Columns: {input_column}")
        print(f"Output Column: {output_column}")
        print(f"Metric: {metric}")
        print(f"Score: {score}")
    
    return score


# Define the hyperparameter tuning function
def hyperparameter_tuning(config, args, verbose=0):
    # class MyHyperModel(kt.HyperModel):
    #     def build(self, hp):
    #         return build_model(hp, config, verbose=verbose)

    #     def fit(self, hp, model, *args, **kwargs):
    #         scores = []
    #         with ProcessPoolExecutor(max_workers=args.executions_per_trial) as executor:  # parallelize executions per trial
    #             futures = [executor.submit(parallel_cross_val_score, hp, config, verbose=verbose) for _ in range(3)]
    #             for future in as_completed(futures):
    #                 scores.append(future.result())
    #         mean_score = np.mean(scores)
    #         # Return the score as a dictionary with a key matching the objective name
    #         return {'val_loss': mean_score}

    hypermodel = MyHyperModel(config, args, verbose)

    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_loss',
        overwrite=True,
        max_trials=args.max_trials,
        executions_per_trial=1,  # Sequential execution
        directory=args.directory,
        project_name=args.project_name
    )

    # Placeholder data required by the search function, it won't be used
    dummy_data = np.zeros((1, 1))
    tuner.search(dummy_data, dummy_data)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

if __name__ == "__main__":
    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "pearson_r2": pearson_r2,
        "hellinger_distance": hellinger_distance,
        "mean_error": mean_error
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("model_type", default=None, help="Model type to be tuned")
    parser.add_argument("predictand_type", default=None, help="Variable we want to predict in this tuner")
    parser.add_argument("-e", "--executions_per_trial", type=int, default=3)
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbose level for debugging")
    parser.add_argument("--directory", type=str, default="my_dir")
    parser.add_argument("--project_name", type=str, default="hyperparam_tuning_")
    parser.add_argument("--max_trials", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    if args.model_type is None:
        print('No model type was given. Search will be cancelled.')
        sys.exit()

    elif args.predictand_type is None:
        print('No predictand type was given. Search will be cancelled.')
        sys.exit()

    # Define model classes
    # model_classes = {
    #     "random_forest": RandomForestRegressor,
    #     "neural_network": DenseNeuralNetwork
    # }

    # Assuming config is defined and contains necessary keys
    best_hyperparameters = hyperparameter_tuning(config, args, verbose=args.verbose)

    # Print best hyperparameters
    print(f"Best Hyperparameters: {best_hyperparameters}")
