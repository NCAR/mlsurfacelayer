import pandas as pd
import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

import plotly
import chart_studio.plotly as py
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
from  plotly.graph_objs import *

import os

from pathlib import Path

import argparse
import yaml

def draw_loss(cut, direc, show=False, save=True, save_path=None):
    if save_path==None: 
        print('Error, save_path is None, cannot save files') 
        return
    
    if not os.path.exists(save_path):
        print(f'Error, save_path "{save_path}" does not exist')
        return

    model_types = [pred1, pred2]

    fig, axs = plt.subplots(1, len(model_types), figsize = (20,10))
    ax = axs.flatten()
    for i in range(len(ax)):
        df = pd.read_csv(direc + '/' + model_types[i] + '-neural_network_training_history.csv')

        ax[i].plot(df[cut:].index, df['loss'][cut:], label='loss')
        ax[i].plot(df[cut:].index, df['val_loss'][cut:], label='val_loss')
        ax[i].set_title('{0} loss function'.format(model_types[i]), fontsize=22)
        ax[i].legend(fontsize=22)

    fig.tight_layout()
    if show: plt.show()
    if save: plt.savefig(f'{save_path}/loss{cut}.eps', format='eps')
    if save: plt.savefig(f'{save_path}/loss{cut}.png', format='png')
    plt.close()
    print("\nfinished loss")

def draw_mf(df, show=False, save=True, save_path=None):
    pred1 = 'momentum_flux'
    ex_pred1 = "momentum_flux:18.4_m:m2_s-2"

    pred2 = 'heat_flux'
    ex_pred2 = 'heat_flux:18.4_m:degrees_C_m_s-1' # degrees celsius , meters per second
    
    mo_version = 'MOST_'



    if save_path==None: 
        print('Error, save_path is None, cannot save files') 
        return   
    
    if not os.path.exists(save_path):
        print(f'Error, save_path "{save_path}" does not exist')
        return
    
    #
    # Create some scatter plots for quick visual comparison of ML models and also MOST calcs
    #
    xStr = ex_pred1
    yStr = []

    yStr.append(pred1 + '-neural_network')
    yStr.append(pred1 + '-random_forest')
    
    yStr.append(mo_version + ex_pred1)
    yStr.append('MOST_chopped_' + ex_pred1)
    yStr.append('MOST_rounded_' + ex_pred1)
    #yStr.append(predictand1+ mo_version2)

    # df2 =  df.loc[df[mo_version + ex_pred1].isna() == False]
    df2 = df[df[mo_version + ex_pred1].notna()]

    #df2 =  df.loc[df[predictand1 + mo_version2].isna() == False]

    fig, ax = plt.subplots(3,2, figsize = (20,20))
    ax = ax.flat
    for i in range (0,len(ax)):
        if i >= len(yStr): continue

        x = df2[xStr]
        y = df2[yStr[i]]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        im = ax[i].scatter(x,y,  c=z )

        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
        fig.colorbar(im, ax = ax[i])
        titleStr = "y = {0} x = measured {1}". format(yStr[i], xStr)
        ax[i].set_title(titleStr )
        m, b = np.polyfit(x, y, 1)
        ax[i].plot(x, x)

        #ax[i].set_xlim(-2.5,1)
        #ax[i].set_ylim(-2.5,1)
        ax[i].set_xlim(-.1,1)
        ax[i].set_ylim(-.1,1)
    
    fig.tight_layout()
    if show: plt.show()
    if save: plt.savefig(save_path + '/mf_scatter.eps', format='eps')
    if save: plt.savefig(save_path + '/mf_scatter.png', format='png')

    plt.close()
    print("\nfinished mf")

def draw_hf(df, show=False, save=True, save_path=None):
    pred1 = 'momentum_flux'
    ex_pred1 = "momentum_flux:18.4_m:m2_s-2"

    pred2 = 'heat_flux'
    ex_pred2 = 'heat_flux:18.4_m:degrees_C_m_s-1' # degrees celsius , meters per second
    
    mo_version = 'MOST_'



    if save_path==None: 
        print('Error, save_path is None, cannot save files') 
        return
    
    if not os.path.exists(save_path):
        print(f'Error, save_path "{save_path}" does not exist')
        return
    
    #   
    # Create some scatter plots for quick visual comparison of ML models and also MOST calcs
    #
    xStr = ex_pred2

    yStr = []

    yStr.append(pred2+ '-neural_network')
    yStr.append(pred2+ '-random_forest')
    
    yStr.append(mo_version + ex_pred2.replace('C','K'))
    yStr.append('MOST_chopped_' + ex_pred2.replace('C','K'))
    yStr.append('MOST_rounded_' + ex_pred2.replace('C','K'))
    #yStr.append(predictand2+ mo_version2)

    # df2 = df.loc[df[mo_version + ex_pred2.replace('C','K')].isna() == False]
    df2 = df[df[mo_version + ex_pred2.replace('C','K')].notna()]

    #df2 = df.loc[df[predictand2+ mo_version2].isna() == False]

    # fig, ax = plt.subplots(2,2, figsize = (20,20))
    fig, ax = plt.subplots(3,2, figsize = (20,20))
    ax = ax.flat
    for i in range (0,len(ax)):
        if i >= len(yStr): continue

        x = df2[xStr]
        y = df2[yStr[i]]

        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
 
        im = ax[i].scatter(x,y,  c=z, cmap = 'plasma')
    
        # ax[i].set_xlim(-.25,.25)
        # ax[i].set_ylim(-.25,.25)
        ax[i].set_xlim(-.5,.5)
        ax[i].set_ylim(-.5,.5)
        # ax[i].set_xlim(-1,1)
        # ax[i].set_ylim(-1,1)
        # ax[i].set_xlim(-1000,1000)
        # ax[i].set_ylim(-1000,1000)
        
        fig.colorbar(im, ax = ax[i])
        titleStr = "x = measured {0}, y = {1}". format(xStr, yStr[i])
        ax[i].set_title(titleStr )
        ax[i].plot(x, x)
     
    fig.tight_layout()
    if show: plt.show()
    if save: plt.savefig(save_path + '/hf_scatter.eps', format='eps')
    if save: plt.savefig(save_path + '/hf_scatter.png', format='png')

    plt.close()
    print("\nfinished hf")

def drawTimeSeries(ax, predictions, exact_predictand, predictand, model_type="neural_network", colorPred='orange'):
    x = predictions['Time']

    y_true=  exact_predictand
    y_pred = predictand + '-' + model_type
    #if model_type == 'MOST_':
    if model_type.startswith('MOST_'):
        y_pred = model_type + exact_predictand.replace('C', 'K')

    y_true = predictions[y_true]
    y_pred = predictions[y_pred]

    ax.plot(x,y_true, color='black', label=predictand + 'true')
    ax.plot(x,y_pred, color=colorPred, label=predictand + 'pred')
    # ax.scatter(x,y_true, color='black', label=predictand + 'true')
    # ax.scatter(x,y_pred, color=colorPred, label=predictand + 'pred')
       
    titleStr = "{0} : {1} ".format(predictand, model_type)
    ax.set_title(titleStr )

    return ax

def draw_group_time_series(df,ex_pred,pred, show=False, save=True, save_path=None):
    if save_path==None: 
        print('Error, save_path is None, cannot save files') 
        return
    
    if not os.path.exists(save_path):
        print(f'Error, save_path "{save_path}" does not exist')
        return
    
    plot_obs = [
        (df, ex_pred, pred, 'neural_network', 'purple'),
        (df, ex_pred, pred, 'random_forest', 'red'),
        (df, ex_pred, pred, 'MOST_', 'green'),
        (df, ex_pred, pred, 'MOST_chopped_', 'orange'),
        (df, ex_pred, pred, 'MOST_chopped_', 'blue')
        #(df, ex_pred, pred, 'mo_branko', 'green'),
        #(df, ex_pred, pred, 'mo_alternate', 'brown'),
    ]

    fig, axs = plt.subplots(2,2, figsize=(20,20))
    ax = axs.flat

    for ax, (p, ep, pr, mt, c) in zip(axs.flat, plot_obs):
        drawTimeSeries(ax, p, ep, pr, model_type=mt, colorPred=c)

    fig.tight_layout()
    if show: plt.show()
    if save: plt.savefig(f'{save_path}/time_series_{pred}.eps', format='eps')
    if save: plt.savefig(f'{save_path}/time_series_{pred}.png', format='png')

    plt.close()
    print("\nfinished time series")

def plot_feature_importance(directory, model_name, regime=None, sort=False, show=False, save=True, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))  # Add this line before the plotting line in both functions
    # plt.tight_layout()


    # Construct file path based on model name and regime if provided
    file_path = f"{directory}/{model_name}_importances.csv"
    if regime:
        file_path = f"{directory}/{model_name}_{regime}_importances.csv"
    
    # Read the CSV file
    imp = pd.read_csv(file_path)
    
    # Calculate average importance
    columns_to_average = [f'all_{i}' for i in range(5)]
    imp['aveImp'] = abs(imp[columns_to_average].mean(axis=1))

    # Sort values by average importance
    bar_plot_imp = imp[['input', 'aveImp']]#.sort_values(by='aveImp')
    
    if sort:
        bar_plot_imp = imp[['input', 'aveImp']].sort_values(by='aveImp')
    else:
        bar_plot_imp = imp[['input', 'aveImp']]#.sort_values(by='aveImp')

    # Plot feature importance
    title = f'Feature Importances for {model_name}'
    ax = bar_plot_imp.plot.barh(color='blue', x='input', y='aveImp', title=title)   

    # ax.set_xticks([0, .005, .01, .015, .02, .025, .03, .035, .04])
    ax.legend(loc='upper right', fontsize='small')

    if show: plt.show()
    if save: plt.savefig(f'{save_path}/{title}_{sort}.eps', format='eps', bbox_inches='tight')
    if save: plt.savefig(f'{save_path}/{title}_{sort}.png', format='png', bbox_inches='tight')
    plt.close()

def average_importances(directory, model_name, sort=False, show=False, save=True, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))  # Add this line before the plotting line in both functions
    # plt.tight_layout()


    # Placeholder for storing feature importances
    feature_importances = []

    # Loop through each fold
    for fold in range(num_folds):
        file_path = directory + f"/model_QC_--kfold-{fold}"
        file_path = f"{file_path}/{model_name}_importances.csv"
        df = pd.read_csv(file_path)
        feature_importances.append(df[['all_0', 'all_1', 'all_2', 'all_3', 'all_4']])

    # Convert the list of DataFrames to a single DataFrame
    all_importances = pd.concat(feature_importances)

    # Compute the average importance
    average_importances = all_importances.groupby(all_importances.index).mean().mean(axis=1)

    # Create a DataFrame for plotting
    if sort:
        imp_df = pd.DataFrame({
        'input': df['input'],  # assuming the input features are the same in all folds
        'aveImp': np.abs(average_importances)
        }).sort_values(by='aveImp')
    else:
        imp_df = pd.DataFrame({
        'input': df['input'],  # assuming the input features are the same in all folds
        'aveImp': np.abs(average_importances)
        })#.sort_values(by='aveImp')

    # Plot the results
    title = f'Feature Importances across {num_folds} K-folds for {model_name}'
    ax = imp_df.plot.barh(color='blue', x='input', y='aveImp', title=title)
    # ax.set_xticks([0, .005, .01, .015, .02, .025, .03, .035, .04])

    if show: plt.show()

    # base_directory = directory.replace(f"/model_QC_--kfold-{}", "")
    if save: plt.savefig(f'{save_path}/{title}_{sort}.eps', format='eps', bbox_inches='tight')
    if save: plt.savefig(f'{save_path}/{title}_{sort}.png', format='png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    #
    # Parse program args:  config file path 
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    #parser.add_argument("--save_file", help="Where to save the graphics")
    parser.add_argument("--show", action='store_true', help='Include this flage to set show to False')
    parser.add_argument("--no-show", action='store_false', dest='show', help='Include this flage to set show to False')
    
    parser.add_argument("--save", action='store_true', help='Include this flage to set save to True')
    parser.add_argument("--no-save", action='store_false', dest='save', help='Include this flage to set save to False')

    parser.add_argument('--average', action='store_true', help='Include this flag to set average to True.')
    parser.add_argument('--no-average', action='store_false', dest='average', help='Include this flag to set average to False.')

    parser.set_defaults(average=False, save=True, show=False)

    parser.add_argument("-v", type=int, default=0)
    #parser.add_argument("--exp_name", type=str, default='model')
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
    
    directory = config['out_dir']
    args.save_file = directory
    
    # model options below ---------------------------------------------------
    pred1 = 'momentum_flux'
    ex_pred1 = "momentum_flux:18.4_m:m2_s-2"

    pred2 = 'heat_flux'
    ex_pred2 = 'heat_flux:18.4_m:degrees_C_m_s-1' # degrees celsius , meters per second
    
    mo_version = 'MOST_'
    #mo_version = '-mo_branko'
    #mo_version2 = '-mo_alternate'

    num_folds = config['k_fold_cross_validation']['N']
    # model options above ---------------------------------------------------
    if args.v >= 1: print(f'this is args.average{args.average} of type {type(args.average)}')
    if args.average == True:
        if args.v >= 1: print('starting average graphs')
        # Define the base directory and number of models
        base_directory = directory.replace(f"/model_QC_--kfold-{config['k_fold_cross_validation']['k']}", "")
        num_models = num_folds  # Adjust based on the number of models

        if args.v >= 1: print(f'base directory {base_directory}')

        # Initialize an empty list to store individual DataFrames
        dfs = []

        # Loop through each model directory
        for i in range(num_models):
            if args.v >= 1: print(f"in loop iter {i}")
            new_directory = f"{base_directory}/model_QC_--kfold-{i}"
            file_path = os.path.join(new_directory, "surface_layer_model_predictions.csv")
            
            if os.path.exists(file_path):
                if args.v >= 1: print(f'inside if {file_path}')
                df = pd.read_csv(file_path) # Load the CSV file into a DataFrame
                dfs.append(df) # Append the DataFrame to the list
            else:
                print(f"File not found: {file_path}")

        # Combine all DataFrames into one
        combined_df = pd.concat(dfs)
        if args.v >= 1: print(f'finished combining {len(combined_df)}')
        
        sort = True
        average_importances(base_directory, f"{pred1}_neural_network", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred2}_neural_network", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred1}_random_forest", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred2}_random_forest", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        print("Averaged Feature Importances sorted created successfully.")

        sort = False
        average_importances(base_directory, f"{pred1}_neural_network", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred2}_neural_network", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred1}_random_forest", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        average_importances(base_directory, f"{pred2}_random_forest", sort=sort, show=args.show, save=args.save, save_path=base_directory)
        print("Averaged Feature Importances unsorted created successfully.")
        
        draw_hf(combined_df, show=args.show, save=args.save, save_path=base_directory)
        draw_mf(combined_df, show=args.show, save=args.save, save_path=base_directory)
        print("Combined DataFrame created successfully.")

    else:
        if args.v >= 1: print('starting single graphs')
        num = config['k_fold_cross_validation']['k']
        sort = True
        plot_feature_importance(f"{directory}", f"{pred1}_neural_network", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred2}_neural_network", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred1}_random_forest", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred2}_random_forest", sort=sort, show=args.show, save=args.save, save_path=directory)
        
        sort = False
        plot_feature_importance(f"{directory}", f"{pred1}_neural_network", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred2}_neural_network", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred1}_random_forest", sort=sort, show=args.show, save=args.save, save_path=directory)
        plot_feature_importance(f"{directory}", f"{pred2}_random_forest", sort=sort, show=args.show, save=args.save, save_path=directory)
        print('finished single feature importances')

        pred = pd.read_csv(directory + "/surface_layer_model_predictions.csv")
        pred['Time'] = pd.to_datetime(pred['Time'])

        # re order the dataframe in chronological order (gets scrambled during training)
        pred = pred.sort_values('Time', ascending=True)
        pred.index = pred['Time'] 

        draw_loss(0, directory, show=args.show, save=args.save, save_path=directory)
        draw_loss(20, directory, show=args.show, save=args.save, save_path=directory)
        draw_hf(pred, show=args.show, save=args.save, save_path=directory)
        draw_mf(pred, show=args.show, save=args.save, save_path=directory)
        draw_group_time_series(pred, ex_pred1, pred1, show=args.show, save=args.save, save_path=directory)
        draw_group_time_series(pred, ex_pred2, pred2, show=args.show, save=args.save, save_path=directory)
        print("\nfinished drawing\n")