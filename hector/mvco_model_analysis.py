
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





def draw_loss(cut, direc):
    model_types = [predictand1, predictand2]

    fig, axs = plt.subplots(1, len(model_types), figsize = (20,10))
    ax = axs.flatten()
    for i in range(len(ax)):
        df = pd.read_csv(direc + '/' + model_types[i] + '-neural_network_training_history.csv')
        # print(df)
        # ax.plot()

        ax[i].plot(df[cut:].index, df['loss'][cut:], label='loss')
        ax[i].plot(df[cut:].index, df['val_loss'][cut:], label='val_loss')
        ax[i].set_title('{0} loss function'.format(model_types[i]), fontsize=22)
        ax[i].legend(fontsize=22)

    plt.show()


def draw_mf(df):
    #
    # Create some scatter plots for quick visual comparison of ML models and also MOST calcs
    #


    xStr=  exact_predictand1

    yStr = []

    yStr.append(predictand1 + '-neural_network')
    yStr.append(predictand1 + '-random_forest')
    # yStr.append('MOSTustar')
    yStr.append(predictand1+ mo_version)
    yStr.append(predictand1+ mo_version2)


    # pred2 = pred.loc[pred['MOSTustar'].isna() == False]
    df2 =  df.loc[df[predictand1 + mo_version].isna() == False]
    df2 =  df.loc[df[predictand1 + mo_version2].isna() == False]

    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2,2, figsize = (20,20))
    ax = ax.flat
    for i in range (0,len(ax)):
        x = df2[xStr]
        y = df2[yStr[i]]
            
        # if yStr[i] == predictand1 + '-neural_network':
        #     y = pred2[yStr[i]]*scaleUstar + meanUstar

        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]


        im = ax[i].scatter(x,y,  c=z )
        #ax[i].axis('equal')
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
        # ax[i].set_xticks([0,.2,.4,.6,.8,1.0])
        # ax[i].set_yticks([0,.2,.4,.6,.8,1.0])
        fig.colorbar(im, ax = ax[i])
        titleStr = "y = {0} x = measured {1}". format(yStr[i], xStr)
        ax[i].set_title(titleStr )
        m, b = np.polyfit(x, y, 1)
        ax[i].plot(x, x)

        ax[i].set_xlim(-2.5,1)
        ax[i].set_ylim(-2.5,1)


def draw_hf(df):
    #
    # Create some scatter plots for quick visual comparison of ML models and also MOST calcs
    #

    xStr = exact_predictand2

    yStr = []

    yStr.append(predictand2+ '-neural_network')
    yStr.append(predictand2+ '-random_forest')
    # yStr.append('TempScaleMOST')
    yStr.append(predictand2+ mo_version)
    yStr.append(predictand2+ mo_version2)


    df2 = df.loc[df[predictand2+ mo_version].isna() == False]
    df2 = df.loc[df[predictand2+ mo_version2].isna() == False]

    # pred2 = pred2.loc[pred2['TempScaleMOST'].isna() == False]
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2,2, figsize = (20,20))
    fig, ax = plt.subplots(2,2, figsize = (20,20))
    ax = ax.flat
    for i in range (0,len(ax)):
        
        x = df2[xStr]
        y = df2[yStr[i]]
            
        # if yStr[i] == predictand2+ '-neural_network':
        #     y =   pred2[yStr[i]]* scaleTscale + meanTscale,
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        #idx = z.argsort()
        #x, y, z = x[idx], y[idx], z[idx]


        im = ax[i].scatter(x,y,  c=z, cmap = 'plasma')
    
        ax[i].set_xlim(-.2,.4)
        ax[i].set_ylim(-.2,.4)
        fig.colorbar(im, ax = ax[i])
        titleStr = "x = measured {0}, y = {1}". format(xStr, yStr[i])
        ax[i].set_title(titleStr )
        #m, b = np.polyfit(x, y, 1)
        ax[i].plot(x, x)

        # ax[i].set_xlim(-.2,.2)
        # ax[i].set_ylim(-1000,1000)


def drawTimeSeries(ax, predictions, exact_predictand, predictand, model_type="neural_network", colorPred='orange'):
    
   
    # x = predictions['Time'].astype(str)
    x = predictions['Time']
    # x = predictions['Time'].index

    y_true=  exact_predictand
    y_pred = predictand + '-' + model_type

    y_true = predictions[y_true]
    y_pred = predictions[y_pred]

    ax.plot(x,y_true, color='black', label=predictand + 'true')
    ax.plot(x,y_pred, color=colorPred, label=predictand + 'pred')
       
    titleStr = "{0} : {1} ". format(predictand, model_type)
    ax.set_title(titleStr )

    return ax


def draw_group_time_series(df,ex_pred,pred):
    plot_obs = [
        (df, ex_pred, pred, 'neural_network', 'purple'),
        (df, ex_pred, pred, 'random_forest', 'red'),
        (df, ex_pred, pred, 'mo_branko', 'green'),
        (df, ex_pred, pred, 'mo_alternate', 'brown'),
    ]

    fig, axs = plt.subplots(2,2, figsize=(20,20))
    ax = axs.flat

    for ax, (p, ep, pr, mt, c) in zip(axs.flat, plot_obs):
        drawTimeSeries(ax, p, ep, pr, model_type=mt, colorPred=c)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    import os

    # Path to the main folder
    main_folder = '../hector'

    # List to store subfolders starting with 'model'
    model_folders = []

    # Traverse the main folder
    for subdir in os.listdir(main_folder):
        subdir_path = os.path.join(main_folder, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith('model'):
            model_folders.append(subdir_path)

    # Loop through the list of model folders and assign each path to a variable
    for idx, folder in enumerate(model_folders):
        variable_name = f'model_folder_{idx + 1}'
        globals()[variable_name] = folder
        print(f'{variable_name}: {folder}')


    #
    # After running train_offshore_models_mvco.py ../config/offshore_surface_layer_training_mvco.yml
    # Read in the predictions from the test data
    #

    # variables needed to make this notebook work for the QC MVCO dataset
    directory = '../hector'

    directory = 'model_QC_data_1000e_50patience_1layer'
    directory2 = 'model_QC_data_1000e_50patience_1layer2'
    predictand1 = 'momentum_flux'
    exact_predictand1 = "momentum_flux:18.4_m:m2_s-2"
    predictand2 = 'heat_flux'
    exact_predictand2 = 'heat_flux:18.4_m:degrees_C_m_s-1' # degrees celsius , meters per second
    mo_version = '-mo_branko'
    mo_version2 = '-mo_alternate'


    pred = pd.read_csv(directory + "/surface_layer_model_predictions.csv")
    pred2 = pd.read_csv(directory2 + "/surface_layer_model_predictions.csv")

    pred['Time'] = pd.to_datetime(pred['Time'])
    pred2['Time'] = pd.to_datetime(pred2['Time'])

    pred.index = pred['Time'] 
    pred2.index = pred2['Time'] 


    model_metrics = pd.read_csv(directory + '/surface_layer_model_metrics.csv')
    model_metrics


    draw_loss(0, directory)


    draw_mf(pred2)
    draw_hf(pred)

    draw_group_time_series(pred, exact_predictand1, predictand1)
    draw_group_time_series(pred, exact_predictand2, predictand2)