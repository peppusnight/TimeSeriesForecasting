import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def example_1():
    # Params
    n_in = 6
    n_out = 1
    n_test = 12
    single_output = False

    # load the dataset
    series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0)
    values = series.values

    # plot dataset
    plt.figure()
    plt.plot(values)
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=n_in, n_out=n_out, single_output=single_output)
    # evaluate
    mae, y, yhat = walk_forward_validation(data.values, n_test = n_test, n_out=n_out)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    plt.figure()
    plt.plot(values)
    n_step_ahead_pred = n_in+n_out-1
    plt.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),y, marker='o', c='C1', label='Expected')
    plt.plot(np.arange(data.values.shape[0]-n_test+n_step_ahead_pred,data.values.shape[0]+n_step_ahead_pred),yhat, marker='o', c='r', label='Predicted')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    print('End example 1!')


def example_2():
    # load the dataset
    series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0)
    values = series.values

    # plot dataset
    plt.plot(values)
    plt.show(block=False)

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=6, n_out=2)
    # evaluate
    mae, y, yhat = walk_forward_validation(data.values, 12)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    plt.figure()
    plt.plot(y, label='Expected')
    plt.plot(yhat, label='Predicted')
    plt.legend()
    plt.show(block=False)

    print('End example 2!')

if __name__ == '__main__':

    print('Start')

    example_1()
    example_2()

    print('End!')


