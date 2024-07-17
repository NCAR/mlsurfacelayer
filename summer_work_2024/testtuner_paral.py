# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messagesimport tensorflow as tf
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
import multiprocessing as mp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlsurfacelayer.metrics import mean_error, hellinger_distance, pearson_r2

from tensorflow.keras.layers import Input
import pandas as pd
import os


class MyHyperModel(kt.HyperModel):
    def __init__(self, config, args, verbose=0):
        self.config = config
        self.verbose = verbose
        self.args = args

    def build(self, hp):
        if self.args.verbose >= 2:
            print(f'inside hypermodel build')
        # Build and return the model instance
        return build_model(hp, self.config, self.args, self.verbose)

    # Room here to implement parallelism : of the executions per trial
    def fit(self, hp, model, *args, **kwargs):
        if self.args.verbose >= 2:
            print('\nCalling MyHyperModel.fit() method\n')
        scores = []
        for _ in range(self.args.executions_per_trial):
            score = parallel_cross_val_score(hp, self.config, model, self.args, verbose=self.verbose)
            scores.append(score)
            if self.args.verbose >= 2:
                print(f'Score for current execution: {score}')
        mean_score = np.mean(scores)
        if self.args.verbose >= 2:
            print(f'Mean score across executions: {mean_score}\n')
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
        if args.verbose >= 2:
            print(f'inside build model before load')
        data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                    devVar=config['k_fold_cross_validation']['devVar'],
                                                    N=config['k_fold_cross_validation']['N'],
                                                    k=0,
                                                    holdout_ratio=config['k_fold_cross_validation']['holdout_ratio'],
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
            # "hidden_layers": hp.Choice('hidden_layers', values=[1, 3, 5]),
            # "lr": hp.Choice('lr', values=[1e-2, 1e-3, 1e-4]),
            # "hidden_neurons": hp.Choice('hidden_neurons', values=[16, 32, 64, 128, 256]),
            # "early_stop": hp.Choice('early_stop', values=[True, False])
            # "optimizer": hp.Choice('optimizer', values=['adam', 'sgd']),
            # "loss": hp.Choice('loss', values=['mean_squared_error', 'mean_absolute_error'])

            "hidden_layers": args.combo[0],
            "lr": args.combo[1],
            "hidden_neurons": args.combo[2],
            "early_stop": args.combo[3]
            
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

def parallel_cross_val_score(hp, config, model, args, verbose=0, max_workers=1):
    if args.verbose >= 2: print('Inside parallel_cross_val_score:')
    
    scores = []

    for fold in range(config['k_fold_cross_validation']['N']):
        if args.verbose >= 2: print(f'Fold number:{fold} in parallel_cross_val_score')
        score = cross_val_score(hp, config, model, args, fold, verbose=verbose)
        scores.append(score)
        
    if len(scores) > 0:
        return np.mean(scores)
    else:
        return None  # Handle case where all folds resulted in errors

# Define the cross-validation function
def cross_val_score(hp, config, model, args, fold, verbose=0):
    # model = build_model(hp, config, verbose=verbose)
    output_type = args.predictand_type
    model_type = args.model_type
    input_column = config["input_columns"][output_type]
    output_column = config["output_columns"][output_type]
    metric = 'mean_absolute_error'
    
    if args.verbose >= 2:
        print(f'inside cross val score before load')
    data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                    devVar=config['k_fold_cross_validation']['devVar'],
                                                    N=config['k_fold_cross_validation']['N'],
                                                    k=fold,
                                                    holdout_ratio=config['k_fold_cross_validation']['holdout_ratio'],
                                                    scramble=config['k_fold_cross_validation']['scramble'])

    if model_type == "random_forest":
        model.fit(data["train"][input_column].values, 
                    data["train"][output_column].values)

        pred = model.predict(data["test"][input_column])
        score = args.metrics[metric](data["test"][output_column].values, pred)

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

        score = args.metrics[metric](data["test"][output_column], pred)
    
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
def hyperparameter_tuning(config, args, verbose=0, type='grid', combo=None):
    if combo == None: 
        print(f'error combo == {combo}')
        return
    else:
        args.combo = combo
        args.project_name = args.project_name + '/combo_' + str(combo)
    
    hypermodel = MyHyperModel(config, args, verbose)
    
    strategy = None
    # strategy = keras.distribute.experimental.CentralStorageStrategy()

    if type == 'grid':
        tuner = kt.tuners.GridSearch(
            hypermodel,
            objective='val_loss',
            overwrite=True,
            directory=args.directory,
            project_name=args.project_name,
            distribution_strategy=strategy
        )
    
    else:
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective='val_loss',
            overwrite=True,
            max_trials=args.max_trials,
            executions_per_trial=1,  # Sequential execution
            directory=args.directory,
            project_name=args.project_name,
            distribution_strategy=strategy
        )

    # Placeholder data required by the search function, it won't be used
    dummy_data = np.zeros((1, 1))
    tuner.search_space_summary()
    tuner.search(dummy_data, dummy_data)

    save_trial_results(tuner, args)

    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # return best_hps
    return tuner


def save_trial_results(tuner, args):
    # trials = tuner.oracle.get_best_trials() # Retrieve all trials
    trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)) # Retrieve all trials
    if args.verbose >= 2:
        print(f"Retrieved {len(trials)} trials.")

    trial_data = [] # Create a list to store trial information

    # Collect trial information
    for trial in trials:
        trial_info = {
            "Trial ID": trial.trial_id,
            "Hyperparameters": trial.hyperparameters.values,
            "Score": trial.score,
            "Status": trial.status
        }
        trial_data.append(trial_info)

    df = pd.DataFrame(trial_data)  # Convert to DataFrame
    if args.verbose >= 2:
        print("Converted trial data to DataFrame.")
    
    # Assuming args.directory and args.project_name are defined somewhere in your code
    output_path = f'{args.directory}/{args.project_name}/../trial_results_.csv'

    # Check if the CSV file exists to determine whether to write the header
    file_exists = os.path.exists(output_path)

    # Write the DataFrame to the CSV file, appending if the file exists
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)
    
    if args.verbose >= 2:
        print(f"Saved trial results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("model_type", default=None, help="Model type to be tuned")
    parser.add_argument("predictand_type", default=None, help="Variable we want to predict in this tuner")
    parser.add_argument("-e", "--executions_per_trial", type=int, default=3)
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbose level for debugging")
    parser.add_argument("--directory", type=str, default="../../data/hypertuner_output")
    parser.add_argument("--project_name", type=str, default="hyperparam_tuning_")
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--search_type", type=str, default='grid')
    args = parser.parse_args()

    args.metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "pearson_r2": pearson_r2,
        "hellinger_distance": hellinger_distance,
        "mean_error": mean_error
    }

    if args.verbose >= 2:
        print(f"Arguments: {args}")

    with open(args.config, "r") as config_file: config = yaml.load(config_file, Loader=yaml.FullLoader) # loads in the configs from the yaml

    if args.model_type is None:
        print('No model type was given. Search will be cancelled.')
        sys.exit()

    elif args.predictand_type is None:
        print('No predictand type was given. Search will be cancelled.')
        sys.exit()

    if args.verbose >= 2:
        print(f"Starting hyperparameter tuning with model type: {args.model_type} and predictand type: {args.predictand_type}")


    
    '''--------------------------------------------------------------'''
    hidden_layers = [1, 3, 5]
    hidden_neurons = [16, 32, 64, 128, 256]
    lr = [0.01, 0.001, 0.0001]
    es = [True, False]

    from itertools import product
    search_space_combinations = list(product(hidden_layers, hidden_neurons, lr, es))
    search_space_combinations = search_space_combinations[:9]
    print(search_space_combinations)
    print(len(search_space_combinations))

    print()
    
    parallel = 4
    # Using multiprocessing.Pool to parallelize cross-validation
    with mp.Pool(processes=parallel) as pool:
        # pool.starmap(hyperparameter_tuning, search_space_combinations)
        pool.starmap(hyperparameter_tuning, [(config, args, 0, 'grid', combo) for combo in search_space_combinations])

    '''--------------------------------------------------------------'''

