import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('Data/data/gas.csv')

# Function to add random noise to a value
def add_noise(value, noise_factor=0.05):
    noise = np.random.uniform(-noise_factor, noise_factor)
    if type(value) == str and "+" in value:
        splitted = value.split("+")
        value1, value2 = float(splitted[0]), float(splitted[1])
        return f"{value1 + noise}+{value2 + noise}"
    return float(value) + noise

# Number of desired additional rows
num_additional_rows = 10

# Initialize an empty DataFrame to store the expanded data
expanded_df = pd.DataFrame(columns=df.columns)

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    # Add the original row to the expanded DataFrame
    expanded_df = expanded_df._append(row, ignore_index=True)
    
    # Add additional rows with noise
    for _ in range(num_additional_rows):
        # Create a new row with added noise
        new_row = row.copy()
        new_row['Sensor 1'] = add_noise(new_row['Sensor 1'])
        new_row['Sensor 2'] = add_noise(new_row['Sensor 2'])
        new_row['ppm'] = add_noise(new_row['ppm'])
        
        # Append the new row to the expanded DataFrame
        expanded_df = expanded_df._append(new_row, ignore_index=True)

# Save the expanded DataFrame to a new CSV file
expanded_df.to_csv('Data/data/expanded_data.csv', index=False)
