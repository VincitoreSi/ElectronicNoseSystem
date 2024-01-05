import pandas as pd
import numpy as np

# write function to interpolate data using linear regression (there are no missing values in data we need to get extra points thats why we are using linear regression)
def interpolate_linear_regression(df, num_additional_rows=10):
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
            new_row["Sensor 1"] = add_noise(new_row["Sensor 1"])
            new_row["Sensor 2"] = add_noise(new_row["Sensor 2"])
            new_row["ppm"] = add_noise(new_row["ppm"])
            new_row["GasType"] = row["GasType"]

            # Append the new row to the expanded DataFrame
            expanded_df = expanded_df._append(new_row, ignore_index=True)

    expanded_df.to_csv("Data/data/interpolate_data.csv", index=False)
    return expanded_df

def add_noise(value, noise_factor=0.05):
    noise = np.random.uniform(-noise_factor, noise_factor)
    if type(value) == str and "+" in value:
        splitted = value.split("+")
        value1, value2 = float(splitted[0]), float(splitted[1])
        return f"{value1 + noise}+{value2 + noise}"
    return float(value) + noise

if __name__ == "__main__":
    # Load the expanded data
    df = pd.read_csv("Data/data/gas.csv")
    df = interpolate_linear_regression(df)
    print(df)