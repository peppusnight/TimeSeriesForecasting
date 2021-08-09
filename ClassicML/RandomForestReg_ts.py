import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import *

def example_0():
    # Replcia di https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
    # Funzione di esempio per trainare e validare un algoritmo per con singolo input e
    # singolo output (anche più timestamp ma una sola variabile di output)
    ############## Params START ##############
    n_in = 6 # Number of "previous data" to use as input
    n_ahead = 1 # Number of step ahead to predict
    single_output = False # If True, predicts only the last n_ahead value; else it predicts all the values "in between"
    n_out = 1 if single_output else n_ahead # Number of outputs of the model
    n_test = 12 # Number of test samples to analyze "at the end of the dataset"; used only for walk forawrd cross validation
    filename = 'daily-total-female-births.csv'
    ############## Params End ##############

    # load the dataset
    series = pd.read_csv(filename, header=0, index_col=0)
    values = series.values

    # plot dataset
    plt.figure()
    plt.plot(values, marker='.')
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_ahead, single_output=single_output)
    # evaluate with walk forawrd validation
    mae, y, yhat, model = walk_forward_validation(data.values, n_test = n_test, n_out=n_out)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    f, ax = plt.subplots(n_out,1,sharex=True, sharey=True)
    ax = ax if n_out>1 else [ax]
    yhat = yhat.reshape(yhat.shape[0],1) if n_out<2 else yhat
    for out_pos, a in enumerate(ax):
        a.plot(values)
        n_step_ahead_pred = n_in+n_ahead-1 if single_output else n_in+out_pos
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),y[:,out_pos],
               marker='.', c='C1', label='Expected - n_step pred.= {}'.format(out_pos + 1))
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),yhat[:,out_pos],
               marker='.', c='r', label='Predicted - n_step pred.= {}'.format(out_pos + 1))
        a.legend()
        a.grid()
        plt.show(block=False)

    if single_output:
        y_new = model.predict(data.values[:, :-n_out])
        plt.figure()
        plt.plot(data.values[:, -1], marker='.')
        plt.plot(y_new)
        plt.show(block=False)

    print('End example 0!')

def example_1():
    # Funzione di esempio per trainare e validare un algoritmo per con singolo input e
    # singolo output (anche più timestamp ma una sola variabile di output)
    ############## Params START ##############
    n_in = 6 # Number of "previous data" to use as input
    n_ahead = 2 # Number of step ahead to predict
    single_output = False # If True, predicts only the last n_ahead value; else it predicts all the values "in between"
    n_out = 1 if single_output else n_ahead # Number of outputs of the model
    n_test = 12 # Number of test samples to analyze "at the end of the dataset"; used only for walk forawrd cross validation
    filename = 'daily-total-female-births.csv'
    ############## Params End ##############

    # load the dataset
    series = pd.read_csv(filename, header=0, index_col=0)
    values = series.values

    # plot dataset
    plt.figure()
    plt.plot(values, marker='.')
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_ahead, single_output=single_output)
    # evaluate with walk forawrd validation
    mae, y, yhat, model = walk_forward_validation(data.values, n_test = n_test, n_out=n_out)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    f, ax = plt.subplots(n_out,1,sharex=True, sharey=True)
    ax = ax if n_out>1 else [ax]
    yhat = yhat.reshape(yhat.shape[0],1) if n_out<2 else yhat
    for out_pos, a in enumerate(ax):
        a.plot(values)
        n_step_ahead_pred = n_in+n_ahead-1 if single_output else n_in+out_pos
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),y[:,out_pos],
               marker='.', c='C1', label='Expected - n_step pred.= {}'.format(out_pos + 1))
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),yhat[:,out_pos],
               marker='.', c='r', label='Predicted - n_step pred.= {}'.format(out_pos + 1))
        a.legend()
        a.grid()
        plt.show(block=False)

    if single_output:
        y_new = model.predict(data.values[:, :-n_out])
        plt.figure()
        plt.plot(data.values[:, -1], marker='.')
        plt.plot(y_new)
        plt.show(block=False)

    print('End example 1!')

def example_2():
    # Funzione di esempio per trainare e validare un algoritmo per con singolo input e
    # singolo output (anche più timestamp ma una sola variabile di output)
    # per la validazione uso il k_fold
    ############## Params START ##############
    n_in = 6
    n_ahead = 1
    single_output = True
    n_out = 1 if single_output else n_ahead
    n_splits = 10 # used only for
    filename = 'daily-total-female-births.csv'
    ############## Params End ##############

    # load the dataset
    series = pd.read_csv(filename, header=0, index_col=0)
    values = series.values

    # plot dataset
    plt.figure()
    plt.plot(values, marker='.')
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_ahead, single_output=single_output)
    # evaluate
    mae, y, yhat, model = k_fold_cross_validation(data.values, n_splits = n_splits, n_out=n_out)
    print('MAE: %.3f' % mae)

    if single_output:
        y_new = model.predict(data.values[:, :-1])
        plt.figure()
        plt.plot(data.values[:, -1], marker='.')
        plt.plot(y_new, marker='.')
        plt.show(block=False)

    print('End example 2!')

def example_3():
    # Funzione di esempio per trainare e validare un algoritmo per con multi-input e
    # singolo output (anche più timestamp ma una sola variabile di output)
    # Params
    n_in = 6
    n_ahead = 12
    n_test = 7*24
    single_output = True
    train_size = 24*365*2
    n_out = 1 if single_output else n_ahead
    filename = 'weather_dataset.csv'
    features_tag = ['Day sin', 'Day cos', 'Year sin','Year cos']
    target_tag = 'T (degC)'

    # load the dataset
    df_orig = pd.read_csv(filename, header=0, index_col=0)
    df_orig = df_orig[ features_tag + [target_tag]]
    df = df_orig.iloc[0:train_size]
    values = df.values

    # plot dataset
    plt.figure()
    plt.plot(df[target_tag].values, marker='.')
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_ahead, single_output=single_output)
    # evaluate
    mae, y, yhat, model = walk_forward_validation(data.values, n_test = n_test, n_out=n_out)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    f, ax = plt.subplots(n_out,1,sharex=True, sharey=True)
    ax = ax if n_out>1 else [ax]
    yhat = yhat.reshape(yhat.shape[0],1) if n_out<2 else yhat
    for out_pos, a in enumerate(ax):
        a.plot(df[target_tag].values)
        n_step_ahead_pred = n_in+n_ahead-1 if single_output else n_in+out_pos
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),y[:,out_pos],
               marker='.', c='C1', label='Expected - n_step pred.= {}'.format(out_pos + 1))
        a.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),yhat[:,out_pos],
               marker='.', c='r', label='Predicted - n_step pred.= {}'.format(out_pos + 1))
        a.legend()
        a.grid()
        plt.show(block=False)

    if single_output:
        all_data = series_to_supervised(df_orig.iloc[train_size:], n_in=n_in, n_out=n_ahead, single_output=single_output)
        y_new = model.predict(all_data.values[:, :-1])
        print('MAE: %.3f' % mean_absolute_error(all_data.values[:, -1], y_new))
        plt.figure()
        plt.plot(all_data.values[:, -1], marker='.')
        plt.plot(y_new,marker='.')
        plt.show(block=True)

    print('End example 2!')

def example_4():
    # Funzione di esempio per trainare e validare un algoritmo per con multi-input e
    # singolo output (anche più timestamp ma una sola variabile di output)
    # Params
    n_in = 6
    n_ahead = 12
    single_output = True
    train_size = 24*365*2
    n_splits = 10
    n_out = 1 if single_output else n_ahead
    filename = 'weather_dataset.csv'
    features_tag = ['Day sin', 'Day cos', 'Year sin','Year cos']
    target_tag = 'T (degC)'

    # load the dataset
    df_orig = pd.read_csv(filename, header=0, index_col=0)
    df_orig = df_orig[ features_tag + [target_tag]]
    df = df_orig.iloc[0:train_size]
    values = df.values

    # plot dataset
    plt.figure()
    plt.plot(df[target_tag].values, marker='.')
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_ahead, single_output=single_output)
    # evaluate
    mae, y, yhat, model = k_fold_cross_validation(data.values, n_splits = n_splits, n_out=n_out)
    print('MAE: %.3f' % mae)

    if single_output:
        all_data = series_to_supervised(df_orig.iloc[train_size:], n_in=n_in, n_out=n_ahead, single_output=single_output)
        y_new = model.predict(all_data.values[:, :-1])
        print('MAE: %.3f' % mean_absolute_error(all_data.values[:, -1], y_new))
        f, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(all_data.values[:, -1], marker='.')
        ax[0].plot(y_new,marker='.')
        ax[1].plot(all_data.values[:, -1]-y_new,marker='.')
        plt.show(block=False)

    print('End example 4!')

if __name__ == '__main__':

    print('Start')

    example_0()
    example_1()
    example_2()
    example_3()
    example_4()


    print('End!')


