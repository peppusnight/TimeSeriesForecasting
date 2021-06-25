import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, single_output=False, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	n_init = n_out-1 if single_output else 0
	for i in range(n_init, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, n_out=1):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-n_out], test[i, -n_out:]
		# fit model on history and make a prediction
		yhat, model = random_forest_forecast(history, testX, n_out=n_out)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected={}, predicted={}'.format(testy, yhat))
	# estimate prediction error
	predictions = np.array(predictions)
	error = mean_absolute_error(test[:, -n_out:], predictions)
	return error, test[:, -n_out:], predictions, model

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX, n_out = 1):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-n_out], train[:, -n_out:]
	# fit model
	clf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
	if n_out<2:
		clf.fit(trainX, trainy.flatten())
		model = clf
	else:
		model = MultiOutputRegressor(clf).fit(trainX, trainy)

	# make a one-step prediction
	yhat = model.predict([testX])
	# model.predict(testX.reshape(1,6))
	return yhat[0], model