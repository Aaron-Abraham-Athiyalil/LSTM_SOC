import numpy as np
import pandas as pd

# Number of readings
num_readings = 10000

# Generate synthetic voltage data (range typically between 40V to 55V)
voltage_data = np.random.uniform(low=40.0, high=55.0, size=num_readings)

# Generate synthetic current data (range typically between -30A to 30A)
current_data = np.random.uniform(low=-30.0, high=30.0, size=num_readings)

# Initialize SoC values starting from a random initial state
initial_soc = np.random.uniform(20, 80)  # Initial SoC between 20% to 80%
soc_data = [initial_soc]

# Battery capacity in Ah
battery_capacity_ah = 30

# Time interval in hours (assuming readings are taken every second)
time_interval_hr = 1 / 3600

# Calculate SoC based on current data
for i in range(1, num_readings):
    # Change in SoC (Delta SoC) = (Current * Time Interval) / Battery Capacity
    delta_soc = (current_data[i-1] * time_interval_hr) / battery_capacity_ah
    new_soc = soc_data[-1] - delta_soc  # Discharging reduces SoC, charging increases it

    # Ensure SoC remains within 0% to 100%
    new_soc = max(0, min(100, new_soc))
    soc_data.append(new_soc)

# Create a DataFrame
data = pd.DataFrame({
    'Voltage (V)': voltage_data,
    'Current (A)': current_data,
    'SoC (%)': soc_data
})

# Save to CSV
data.to_csv('synthetic_battery_soc_data.csv', index=False)

print("Synthetic dataset with SoC created successfully!")
