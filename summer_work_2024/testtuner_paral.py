# import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messagesimport tensorflow as tf
from tensorflow import keras
from itertools import product
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
from mlsurfacelayer.models import save_random_forest_csv, save_scaler_csv

from tensorflow.keras.layers import Input
import pandas as pd
import os

import warnings

warnings.filterwarnings('ignore', message='The `lr` argument is deprecated', category=UserWarning)

#
# MyHyperModel class is a custom implementation for hyperparameter tuning using Keras Tuner. It allows:
#  -Building a model with a given hyperparameter configuration.
#  -Custom fitting logic that includes parallel cross-validation.
#
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

#
# Model builder: either random forest or neural network
#
def build_model(hp, config, args, verbose=0):
    
    #
    # model_type = 'random_forest' or 'neural_network
    #
    model_type = args.model_type
   
    #
    # the hyperparams for the model
    # 
    model_config = config["model_config"][model_type]
    
    if model_type == "random_forest":
        if args.verbose >= 2:
            model_config.update({
                "verbose": 2
            })

        if args.search_type == 'grid':
            model_config.update({
                "n_estimators": args.num_estimators,
                "max_depth": args.maximum_depth,
                "n_leaf_nodes": args.num_leaf_nodes,
                "max_features": args.maximum_features
            })

        elif args.search_type == 'bay':
            model_config.update({
                "n_estimators": hp.Int('n_estimators', min_value=10, max_value=100, step=10),
                "max_depth": hp.Int('max_depth', min_value=3, max_value=10, step=1)
            })

        elif args.search_type == 'bay_ext':
            model_config.update({
                "n_estimators": hp.Int('n_estimators', min_value=10, max_value=100, step=10),
                "max_depth": hp.Int('max_depth', min_value=3, max_value=10, step=1)
            })
       
        #
        # Instantiate the model
        #
        model = RandomForestRegressor(**model_config)
    
    elif model_type == "neural_network":
        print( "model_config1: ", model_config)
        if args.verbose >= 2:
            print(f'inside build model before load')
        
        if args.verbose >= 2: 
            model_config.update({
                "verbose": 2
            })

        if args.search_type == 'grid':
            model_config.update({
            "hidden_layers": args.hl,
            "lr": args.lr,
            "hidden_neurons": args.hn,
            "early_stop": args.es
            })

        elif args.search_type == 'bay':
            model_config.update({
            "hidden_layers": hp.Choice('hidden_layers', values=[1, 3, 5]),
            "lr": hp.Choice('lr', values=[1e-2, 1e-3, 1e-4]),
            "hidden_neurons": hp.Choice('hidden_neurons', values=[16, 32, 64, 128, 256])
            })

        elif args.search_type == 'bay_ext':
            model_config.update({
            "hidden_layers": hp.Choice('hidden_layers', values=[1, 2, 3, 4, 5]),
            "lr": hp.Choice('lr', values=[1e-2, 1e-3, 1e-4, 1e-5]),
            "hidden_neurons": hp.Choice('hidden_neurons', values=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
            # "early_stop": hp.Choice('early_stop', values=[True, False])
            # "optimizer": hp.Choice('optimizer', values=['adam', 'sgd']),
            # "loss": hp.Choice('loss', values=['mean_squared_error', 'mean_absolute_error'])
            "activation": hp.Choice('activation', values=['tanh', 'relu'])
            })
 
        #
        # Create a model object
        # 
        model = DenseNeuralNetwork(**model_config)

        #
        # Build the model 
        #
        output_type = args.predictand_type
        model = model.build_neural_network( inputs = len(config["input_columns"][output_type]) , outputs = 1 )   
    
    if verbose >= 2:
        print(f"Step: Build Model")
        print(f"Model Class: {model_type}")
        print(f"Model Config: {model_config}")
    
    return model

#
# Note : max_workers not used
#        hp passed to another method but not used in that method
# 
def parallel_cross_val_score(hp, config, model, args, verbose=0, max_workers=1):

    if args.verbose >= 2: print('Inside parallel_cross_val_score:')
    
    scores = []

    #
    # Collect scores from each cross validation fold
    #
    for fold in range(config['k_fold_cross_validation']['N']):
        if args.verbose >= 2: print(f'Fold number:{fold} in parallel_cross_val_score')
        score = cross_val_score(hp, config, model, args, fold, verbose=verbose)
        scores.append(score)
        
    if args.verbose >= 2:
        print("cross validation scores: ", scores)

    if len(scores) > 0:
        print('\nmean of cross validation scores: ',np.mean(scores))
        return np.mean(scores)
    else:
        return None  # Handle case where all folds resulted in errors


# Perform the cross validation 
#
# hp:
#
def cross_val_score(hp, config, model, args, fold, verbose=0):


    print("loading data for fold ", fold)   
    data = load_derived_data_random_test_train_val(config["data_file"], dropna=True, 
                                                   devVar=config['k_fold_cross_validation']['devVar'],
                                                   N=config['k_fold_cross_validation']['N'],
                                                   k=fold,
                                                   holdout_ratio=config['k_fold_cross_validation']['holdout_ratio'],
                                                   scramble=config['k_fold_cross_validation']['scramble'],
                                                   config=config)

    #
    # Get parameters for model traing and fitting: predictors, predictand, metric, and ml model type
    #
    output_type = args.predictand_type
    input_column = config["input_columns"][output_type]
    output_column = config["output_columns"][output_type]
    metric = 'mean_absolute_error'
    model_type = args.model_type

    #
    # Fit the model to the training data, predict on test data,
    # apply metric to truth and predictions
    #
    if model_type == "random_forest":
       
        model.fit(data["train"][input_column].values, 
                    data["train"][output_column].values)

        pred = model.predict(data["test"][input_column])
        score = args.metrics[metric](data["test"][output_column].values, pred)

    elif model_type == "neural_network":
        #
        # Neural network data is scaled using standard z-score
        #
        input_scaler = StandardScaler()
        
        #
        # Scale the predictor and predictand cols of data in the train, validate, and test sets
        #
        var_scale_list = input_column + [output_column]
        scaled_train = input_scaler.fit_transform(data["train"][var_scale_list])
        scaled_val = input_scaler.transform(data["validate"][var_scale_list])
        scaled_test = input_scaler.transform(data["test"][var_scale_list])
 
        #
        # Fit the model to the training data, predict 
        #       
        model.fit(scaled_train[:, :-1], scaled_train[:, -1], scaled_val[:, :-1], scaled_val[:, -1])
        pred = model.predict(scaled_test[:, 0:-1])

        #
        # Predictions are scaled, so unscale and apply metric to truth and predictions 
        #
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
#
# Instantiate the hyperparameter tuner ( grid or bayesian)
#
def hyperparameter_tuning(config, args, verbose=0, type='grid', combo=None):

    if combo == None: 
        print(f'error combo == {combo}')
        return
    else:
        # Descriptive project name: contains all of the hyper params: 
        # eg. for hyper params 5 layers, 128 neurons, learning rate .001 would be  "combo__5__128__0_001__True_" 
        args.project_name = args.project_name_dir + '/combo_' + str(combo).replace('(','_').replace(')','_').replace(' ', '_').replace(',', '_').replace('.','_')
        print('projectname: ', args.project_name)
    
    #
    # Instantiate a place holder hypermodel needed for instantiating
    # keras tuner
    #
    hypermodel = MyHyperModel(config, args, verbose)

    #
    # distributive processing strategy 
    #     
    strategy = None

    #
    # Instantiate a tuner based on 
    #
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
    #
    # In Keras Tuner, the search function requires some form of data to initiate the search process, 
    # even though this data may not be used during the hyperparameter tuning. 
    # The purpose of this placeholder data is to satisfy the function signature and allow the 
    # tuner to go through its workflow of hyperparameter space exploration.
    #
    dummy_data = np.zeros((1, 1))
    tuner.search_space_summary()
    tuner.search(dummy_data, dummy_data)

    save_trial_results(tuner, args)

    return tuner

#
# I dont quite understand : used for grid search only (?)
#
def save_trial_results(tuner, args):
    # 
    # Retrieve all trials
    #
    trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)) 

    if args.verbose >= 2:
        print(f"Retrieved {len(trials)} trials.")

    trial_data = [] # Create a list to store trial information

    # Collect trial information
    for trial in trials:
        trial_info = {
            "Trial ID": trial.trial_id,
            "Hyperparameters": {"hidden_layers": args.hl,"hidden_neurons": args.hn,"learning_rate": args.lr,"early_stop": args.es,},
            "Score": trial.score,
            "Status": trial.status
        }
        trial_data.append(trial_info)

    #
    # Save trial info
    #
    df = pd.DataFrame(trial_data)  # Convert to DataFrame
    if args.verbose >= 2:
        print("Converted trial data to DataFrame.")
    
    # Assuming args.directory and args.project_name are defined somewhere in your code
    # output_path = f'{args.directory}/{args.project_name}/../trial_results_.csv'
    output_path = f'{args.directory}/{args.project_name}/trial_results_.csv'

    # Check if the CSV file exists to determine whether to write the header
    file_exists = os.path.exists(output_path)

    # Write the DataFrame to the CSV file, appending if the file exists
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)

    output_path = f'{args.directory}/{args.project_name}/../trial_results_.csv'

    df.to_csv(output_path, mode='a', header=False, index=False)
    
    if args.verbose >= 2:
        print(f"Saved trial results to {output_path}")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("model_type", default=None, help="Model type to be tuned")
    parser.add_argument("predictand_type", default=None, help="Variable we want to predict in this tuner")
    parser.add_argument("-e", "--executions_per_trial", type=int, default=1)
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbose level for debugging")
    parser.add_argument("--directory", type=str, default="../../data/hypertuner_output")
    parser.add_argument("--project_name", type=str, default="hyperparam_tuning_")
    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--search_type", type=str, default='grid')
    parser.add_argument("-p", type=int, default=4)
    args = parser.parse_args()
    return args

# if __name__ == "__main__":
args = parser()

args.project_name_dir = args.project_name

args.metrics = {
#    "mean_squared_error": mean_squared_error,
     "mean_absolute_error": mean_absolute_error
#    "pearson_r2": pearson_r2,
#    "hellinger_distance": hellinger_distance,
#    "mean_error": mean_error
}

if args.verbose >= 2:
    print(f"Arguments: {args}")

# loads in the configs from the yaml
with open(args.config, "r") as config_file: 
   config = yaml.load(config_file, Loader=yaml.FullLoader)

# type: 'random_forest' or neural network
if args.model_type is None:
    print('No model type was given. Search will be cancelled.')
    sys.exit()

elif args.predictand_type is None:
    print('No predictand type was given. Search will be cancelled.')
    sys.exit()

if args.verbose >= 2:
    print(f"Starting hyperparameter tuning with model type: {args.model_type} and predictand type: {args.predictand_type}")

'''--------------------------------------------------------------'''
# neural network model parameters
hidden_layers = [1, 3, 5]
hidden_neurons = [16, 32, 64, 128, 256]
lr = [0.01, 0.001, 0.0001]
activation = ['tanh', 'relu']
es = [True]

# random forest model parameters
# number of trees in the forest
num_estimators: [100, 200, 500]

#The number of features to consider when looking for the best split at each node. 
#Diversity of Trees:
#Higher Diversity: A smaller max_features value means fewer features are considered for each split, leading to more diverse trees. This diversity helps reduce overfitting and improves generalization.
#Lower Diversity: A larger max_features value means more features are considered for each split, which can reduce the diversity among trees and potentially lead to overfitting.
#Model Performance:
#Accuracy: Tuning max_features can help improve the accuracy of the model. Using too few features might lead to underfitting, while using too many can lead to overfitting.
#Bias-Variance Tradeoff: Smaller values of max_features tend to increase bias but reduce variance, leading to more stable models. Larger values tend to decrease bias but increase variance.
#Computational Efficiency:
#Smaller Values: Reduce the computational burden as fewer features are considered at each split, speeding up the model training.
#Larger Values: Increase computational cost as more features are evaluated at each node split, potentially slowing down the training process.
maximum_features: [2, 3, 5, 10 ]

# max_leaf_nodes in a Random Forest model helps control the complexity of each tree in the forest, 
# which can prevent overfitting. By limiting the maximum number of leaf nodes, you can make the individual 
# trees simpler and more generalizable. 
#
maximum_leaf_nodes: [32,64, 128, 256, 516, 1024]

#
# for grid search, produce the set of parameter combinations
#
search_space_combinations = list(product(hidden_layers, hidden_neurons, lr, es, activation))

# search_space_combinations = [(config, args, 0, 'grid', combo) for combo in search_space_combinations]
print('\n\n\n search:', search_space_combinations, len(search_space_combinations))
parallel = args.p #4

print( args)

def wrapper(combo):
    if args.model_type == 'neural_network':
        args.hl, args.hn, args.lr, args.es, args.act = combo
    elif args.model_type == 'random_forest':
        args.ne, args.mf, args.nj, args.ml = combo
    
    # print(f'inside wrapper: {combo}\n{config}\n{args}')
    print(f'in wrapper {args.hl}, {args.hn}, {args.lr}, {args.es}, {args.act}')
    hyperparameter_tuning(config, args, verbose=0, type='grid', combo=combo)

if __name__ == '__main__':

    # combo=(1,16,0.01,True)
    # wrapper(combo=combo)
    # exit()

    # Using multiprocessing.Pool to parallelize cross-validation
    if args.search_type == 'grid':
        with mp.Pool(processes=parallel) as pool:
            # pool.starmap(wrapper, search_space_combinations)
            pool.map(wrapper, search_space_combinations) 
            # for combo in search_space_combinations:
            #     pool.apply_async(wrapper, args=(combo,))
        '''--------------------------------------------------------------'''
    else:
        # combo is a dummy variable to keep the signature
        combo = (0,0,0,0,0)
        hyperparameter_tuning(config, args, verbose=0, type=args.model_type, combo=combo)
