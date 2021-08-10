import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from WindowGenerator import WindowGenerator

# Load data ####################################################################################
df = pd.read_csv('weather_dataset.csv', index_col=0)

# Split data ####################################################################################
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# Scale data ####################################################################################
# scaler = StandardScaler()
# train_df = scaler.fit_transform(train_df)
# val_df = scaler.transform(val_df)
# test_df = scaler.transform(test_df)

train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Create window of single step ####################################################################################
w1 = WindowGenerator(
    train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=6, label_width=1, shift=1, batch_size=32,
    label_columns=['T (degC)'])

# Simple Linear Model ####################################################################################
linear_ds = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
def compile_and_fit(model, window, patience=2, MAX_EPOCHS = 20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

history = compile_and_fit(linear_ds,w1)


np_ds = w1.from_ds_to_numpy_flatten()
linear_numpy = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2,
                                                  mode='min')

linear_numpy.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

history_np = linear_numpy.fit(x=np_ds['train']['X'], y=np_ds['train']['y'], epochs=20,
                    validation_data = (np_ds['val']['X'], np_ds['val']['y']),
                    callbacks=[early_stopping])


column_indices = {name: i for i, name in enumerate(df.columns)}
num_features = df.shape[1]