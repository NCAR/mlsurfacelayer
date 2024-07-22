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

    df2 =  df.loc[df[mo_version + ex_pred1].isna() == False]
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

    df2 = df.loc[df[mo_version + ex_pred2.replace('C','K')].isna() == False]
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
    
        ax[i].set_xlim(-.25,.25)
        ax[i].set_ylim(-.25,.25)
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

if __name__ == "__main__":
    #
    # Parse program args:  config file path 
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    #parser.add_argument("--save_file", help="Where to save the graphics")
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=True)
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
    # model options above ---------------------------------------------------

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
