import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, concatenate
from keras_tuner import HyperModel, RandomSearch
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
target_scaled = scaler.fit_transform(target.reshape(-1, 1)).flatten()

# Reshape the data for LSTM/GRU (samples, timesteps, features)
timesteps = 10
X, y = [], []
for i in range(len(features_scaled) - timesteps):
    X.append(features_scaled[i:i+timesteps])
    y.append(target_scaled[i+timesteps])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hypermodel
class SOCHyperModel(HyperModel):
    def build(self, hp):
        input_layer = Input(shape=(timesteps, X_train.shape[2]))
        
        # CNN part
        conv1 = Conv1D(
            filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
            kernel_size=3,
            activation='relu'
        )(input_layer)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        flat = Flatten()(pool1)
        
        # LSTM part
        lstm = LSTM(
            units=hp.Int('lstm_units', min_value=50, max_value=200, step=50),
            return_sequences=False
        )(input_layer)
        
        # Concatenate CNN and LSTM outputs
        concat = concatenate([flat, lstm])
        dropout = Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(concat)
        output = Dense(1)(dropout)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                      loss='mean_squared_error')
        
        return model

# Instantiate the hypermodel and tuner
hypermodel = SOCHyperModel()
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='soc_prediction'
)

# Perform the search
tuner.search(X_train, y_train, epochs=30, validation_split=0.2)

# Get the optimal hyperparameters and build the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model
loss = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # De-normalize predictions

# Print actual and predicted values
print("Actual SoC values:", scaler.inverse_transform(y_test.reshape(-1, 1)).flatten())
print("Predicted SoC values:", predictions.flatten())

# Display predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual SoC')
plt.plot(predictions, label='Predicted SoC')
plt.title('SoC Prediction using Tuned CNN-LSTM Model')
plt.xlabel('Sample')
plt.ylabel('SoC (%)')
plt.legend()
plt.show()
