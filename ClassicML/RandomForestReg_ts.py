import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *


def example_1():
    # Params
    n_in = 6
    n_ahead = 1
    n_test = 12
    single_output = True
    n_out = 1 if single_output else n_ahead
    filename = 'daily-total-female-births.csv'

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
        y_new = model.predict(data.values[:, :-1])
        plt.figure()
        plt.plot(data.values[:, -1], marker='.')
        plt.plot(y_new)
        plt.show(block=False)

    print('End example 1!')


def example_2():

    # Params
    n_in = 6
    n_ahead = 12
    n_test = 48
    single_output = True
    n_out = 1 if single_output else n_ahead
    filename = 'weather_dataset.csv'
    features_tag = ['Day sin', 'Day cos'] # ,'Year sin','Year cos']
    target_tag = 'T (degC)'

    # load the dataset
    df_orig = pd.read_csv(filename, header=0, index_col=0)
    df_orig = df_orig[ features_tag + [target_tag]]
    df = df_orig.iloc[0:24*365]
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
        all_data = series_to_supervised(df_orig.iloc[24*365:], n_in=n_in, n_out=n_ahead, single_output=single_output)
        y_new = model.predict(all_data.values[:, :-1])
        print('MAE: %.3f' % mean_absolute_error(all_data.values[:, -1], y_new))
        plt.figure()
        plt.plot(all_data.values[:, -1], marker='.')
        plt.plot(y_new,marker='.')
        plt.show(block=False)

    print('End example 2!')

if __name__ == '__main__':

    print('Start')

    # example_1()
    example_2()

    print('End!')


