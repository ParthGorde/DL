# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set up matplotlib configuration
mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['axes.grid'] = False

# Load the dataset from "ecg_final.txt"
# !cat "/content/ECG5000_TRAIN.txt" "/content/ECG5000_TEST.txt" > ecg_final.txt 
# Note: The dataset is assumed to be in the current working directory
df = pd.read_csv("../ecg_final.txt", sep='  ', header=None, engine='python')

# Check the shape of the dataset
df.shape

# Add prefix 'c' to column names for consistency
df = df.add_prefix('c')

# Check the counts of unique values in the 'c0' column
df['c0'].value_counts()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.values, df.values[:, 0:1], test_size=0.2, random_state=111)

# Initialize a MinMaxScaler to scale the data
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
data_scaled = scaler.fit(x_train)
train_data_scaled = data_scaled.transform(x_train)
test_data_scaled = data_scaled.transform(x_test)

# Extract normal and anomaly data from the scaled training and testing data
normal_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 == 0').values[:, 1:]
anomaly_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:]
normal_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:, 1:]
anomaly_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:]

# Plot normal training data
plt.plot(normal_train_data[0])
plt.plot(normal_train_data[1])
plt.plot(normal_train_data[2])
plt.title("Normal Data")
plt.show()

# Plot anomaly training data
plt.plot(anomaly_train_data[0])
plt.plot(anomaly_train_data[1])
plt.plot(anomaly_train_data[2])
plt.title("Anomaly Data")
plt.show()

# Create an Autoencoder model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(140, activation="sigmoid"))

# Compile the Autoencoder model
model.compile(optimizer='adam', loss='mean_squared_error')
