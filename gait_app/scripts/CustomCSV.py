import numpy as np
import pandas as pd
from mat_to_csv import MatToCsv


def calculate_angular_velocity(angles, delta_t):
    return np.concatenate(([np.nan], np.diff(angles) / delta_t))

def calculate_integral(angles, delta_t):
    return np.trapz(angles, dx=delta_t)

def preprocess_dataset(file_name):
    matToCsv = MatToCsv()
    dataset = matToCsv.dataset(file_name)

    # Extract name and velocity from the file name
    name = 'Shank' if 'Shk' in file_name else 'Thigh'
    velocity = float(file_name.split('_')[1].split('ms')[0]) / 10

    # Melt the dataframe
    melted_df = dataset.melt(ignore_index=False, var_name='subjects', value_name='values')

    # Add velocity column
    melted_df[name + 'Velocity'] = velocity

    # Reset index
    melted_df.reset_index(inplace=True)

    # Rename columns
    melted_df.rename(columns={'index': 'gait_percentage'}, inplace=True)

    # Add 1 to the 'index_original' column to start the index from 1
    melted_df['gait_percentage'] = melted_df['gait_percentage'] + 1

    # angular velocity
    delta_t = 0.01  # provided by team
    angles = melted_df['values'].to_numpy()
    angular_velocity = [calculate_angular_velocity(angles[i:i + 100], delta_t) for i in range(0, len(angles), 100)]
    melted_df[name + 'AngularVelocity'] = np.concatenate(angular_velocity)

    # integral
    integral = [calculate_integral(angles[i:i + 100], delta_t) for i in range(0, len(angles), 100)]
    melted_df[name + 'Integral'] = np.repeat(integral, 100)

    df = melted_df.rename(columns={'values': name + 'Angles'})
    # Reorder the DataFrame columns
    df = df[['gait_percentage', name + 'Angles', name + 'Velocity', name + 'AngularVelocity', name + 'Integral']]

    return df

file_names = ['ShkAngW_05ms.mat', 'ShkAngW_10ms.mat', 'ShkAngW_15ms.mat', 'ThiAngW_05ms.mat', 'ThiAngW_10ms.mat', 'ThiAngW_15ms.mat']
datasets = [preprocess_dataset(file_name) for file_name in file_names]

# DataFrame concatenation
shankDF = pd.concat(datasets[:3], axis=0)
thighDF = pd.concat(datasets[3:], axis=0).drop('gait_percentage', axis=1)
bothDF = pd.concat([shankDF, thighDF], axis=1)

bothDF.to_csv('melted_dataset.csv', index=False)
