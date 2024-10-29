import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Ensure the default encoding is utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the synthetic dataset with utf-8 encoding
data = pd.read_csv('synthetic_battery_soc_data.csv', encoding='utf-8')

# Features and target
features = data[['Voltage (V)', 'Current (A)']].values
target = data['SoC (%)'].values

# Normalize the data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Reshape the data for LSTM/GRU (samples, timesteps, features)
timesteps = 10
X, y = [], []
for i in range(len(features_scaled) - timesteps):
    X.append(features_scaled[i:i+timesteps])
    y.append(target[i+timesteps])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN part
input_layer = Input(shape=(timesteps, X_train.shape[2]))
conv1 = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat = Flatten()(pool1)

# LSTM part
lstm = LSTM(100, return_sequences=False, kernel_regularizer=l2(0.01))(input_layer)

# Concatenate CNN and LSTM outputs
concat = concatenate([flat, lstm])
dropout = Dropout(0.3)(concat)
output = Dense(1, kernel_regularizer=l2(0.01))(dropout)

# Build and compile the model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, 100)  # Ensure predictions are within 0-100%

# Display predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual SoC')
plt.plot(predictions, label='Predicted SoC')
plt.title('SoC Prediction using Optimized CNN-LSTM Model')
plt.xlabel('Sample')
plt.ylabel('SoC (%)')
plt.legend()
plt.show()
