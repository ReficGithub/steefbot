import os
import h5py
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import time
# lr: 0.0001, dr: 0.001, nn: 500, bs: 24
input_shape = (200, 6)
learning_rate = 0.0001
dropout_rate = 0.001
epochs = 10
batch_size = 24
validation_split = 0.2
aantal_aandelen = 3

num_neurons = 750

data_directory = 'made'
data_files = os.listdir(data_directory)
random_files = random.sample(data_files, aantal_aandelen)
lstm_model = None

def custom_loss(y_true, y_pred):
    predicted_low = y_pred[:, 0]
    predicted_high = y_pred[:, 1]
    
    actual_low = y_true[:, 0]
    actual_high = y_true[:, 1]

    loss_low = tf.reduce_mean(tf.abs((actual_low - predicted_low) / actual_low) ** 2)
    loss_high = tf.reduce_mean(tf.abs((actual_high - predicted_high) / actual_high) ** 2)

    penalty_low = tf.reduce_mean(tf.maximum(0.0, (predicted_low - actual_low) / actual_low))
    penalty_high = tf.reduce_mean(tf.maximum(0.0, (actual_high - predicted_high) / actual_high))

    total_loss = ((loss_high + loss_low) + (penalty_high + penalty_low)) * 100

    return total_loss


def create_lstm_model(input_shape, dropout_rate=dropout_rate, num_neurons=num_neurons):
    # Controleer of er al een model bestaat
    if os.path.exists("model.keras"):
        print("Model loaded!")
        model = tf.keras.models.load_model("model.keras")
    else:
        print("Creating model...")
        model = Sequential()
        model.add(LSTM(num_neurons, dropout=dropout_rate, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(num_neurons, dropout=dropout_rate, return_sequences=True))
        model.add(LSTM(num_neurons, dropout=dropout_rate, return_sequences=True))
        model.add(LSTM(num_neurons, dropout=dropout_rate, return_sequences=True))
        model.add(LSTM(num_neurons, dropout=dropout_rate, return_sequences=True))
        model.add(LSTM(num_neurons, dropout=dropout_rate))
        model.add(Dense(2))
    return model

for file_name in random_files:
    file_path = os.path.join(data_directory, file_name)
    Xtrain = []
    ytrain = []
    with h5py.File(file_path, 'r') as f:
        if 'OHLC' in f:
            df = f['OHLC'][:]
            np.random.shuffle(df)
            num_sets, num_rows, num_cols = df.shape
            for i in range(num_sets):
                training_set = df[i]
                min_val = np.min(training_set[:-2, :4])
                max_val = np.max(training_set[:-2, :4])
                min_val_5 = np.min(training_set[:-2, 4])
                max_val_5 = np.max(training_set[:-2, 4])
                normalized_set = np.copy(training_set)
                normalized_set[:, :4] = 1+ ((training_set[:, :4] - min_val) / (max_val - min_val))
                normalized_set[:, 4] = 1+ ((training_set[:, 4] - min_val_5) / (max_val_5 - min_val_5))
                normalized_set[:, -2:] = 1+ ((training_set[:, -2:] - min_val) / (max_val - min_val))
                Xtrain.append(normalized_set[:num_rows, :6])
                ytrain.append(normalized_set[0, -2:])
            print("Dataset created for", file_name)
            Xtrain = np.array(Xtrain)
            ytrain = np.array(ytrain)
            if lstm_model == None:
                lstm_model = create_lstm_model(input_shape)
            optimizer = Adam(learning_rate=learning_rate)
            lstm_model.compile(optimizer=optimizer, loss=custom_loss)
            starttijd = time.time()
            lstm_model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
            eindtijd = time.time()
            tijd = eindtijd - starttijd
            print("Uren:", ((tijd/60)/60), "Minuten:", (tijd/60))

lstm_model.save("model.keras")
print("Model trained and saved!")
