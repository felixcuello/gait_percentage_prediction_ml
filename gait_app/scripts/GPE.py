import numpy as np
import pandas as pd
from mat_to_csv import MatToCsv
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import logging

def calculate_angular_velocity(angles, delta_t):
    return np.concatenate(([np.nan], np.diff(angles) / delta_t))

def calculate_integral(angles, delta_t, velocity):
    integral_values = []
    integral_sum = 0

    for i in range(len(angles) - 1):
        integral_sum += velocity * (angles[i + 1] + angles[i]) * delta_t
        integral_values.append(integral_sum)

    # Add the initial condition (0) to the beginning of the integral_values list
    integral_values.insert(0, 0)

    return integral_values

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
    delta_t = 1  # provided by team
    angles = melted_df['values'].to_numpy()
    angular_velocity = [calculate_angular_velocity(angles[i:i + 100], delta_t) for i in range(0, len(angles), 100)]
    melted_df[name + 'AngularVelocity'] = np.concatenate(angular_velocity)

    #integral
    integral = [calculate_integral(angles[i:i + 100], delta_t, velocity) for i in range(0, len(angles), 100)]
    melted_df[name + 'Integral'] = np.concatenate(integral)  # Updated this line

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

#bothDF.to_csv('melted_dataset.csv', index=False)

# ------------------------------------------------------------------
#   Add Data Agumentation
# ------------------------------------------------------------------
augment_data = True

if augment_data:
    # from data_augmentation import TimeShift [DOES NOT WORK]
    # ts = TimeShift()
    # bothDF = ts.process(bothDF)

    from data_augmentation import Interpolation
    i = Interpolation()
    bothDF = i.process(bothDF)

    from data_augmentation import NoiseInjection
    ni = NoiseInjection()
    bothDF = ni.process(bothDF, noise_variability=3)
    bothDF = ni.process(bothDF, noise_variability=5)

    from data_augmentation import Scaling
    sc = Scaling()
    bothDF = sc.process(bothDF, scale_percent=10)
    bothDF = sc.process(bothDF, scale_percent=-10)

# ------------------------------------------------------------------


# X = bothDF[['ShankAngles','ShankVelocity','ShankAngularVelocity','ShankIntegral','ThighAngles','ThighVelocity','ThighAngularVelocity','ThighIntegral']]
X = bothDF[['ShankAngles','ShankVelocity','ShankAngularVelocity','ThighAngles','ThighVelocity','ThighAngularVelocity']]
y = bothDF['gait_percentage']
bothDF.to_csv('merged_dataset.csv', index=False)

#  In case we're augmenting the data, we will use the "real" data as the test set and
#  the augmented data as the training set
if augment_data:
    test_X = X.iloc[0:6299]
    test_y = y.iloc[0:6299]
    train_X = X.iloc[6300:]
    train_y = y.iloc[6300:]
else:
    # Splitting the whole dataset
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)


# Instantiation
xgb_r = xg.XGBRegressor(objective='reg:gamma',
                        n_estimators=100, seed=123)
# Fitting the model
xgb_r.fit(train_X, train_y)

# Predict the model
pred = xgb_r.predict(test_X)

# RMSE Computation
rmse = np.sqrt(MSE(test_y, pred))

# Training RMSE
train_pred = xgb_r.predict(train_X)
train_rmse = np.sqrt(MSE(train_y, train_pred))
train_r2 = r2_score(train_y, train_pred)

# Test RMSE
test_pred = xgb_r.predict(test_X)
test_rmse = np.sqrt(MSE(test_y, test_pred))
test_r2 = r2_score(test_y, test_pred)

print(f"train r2   => {train_r2}")
print(f"test  r2   => {test_r2}")
print(f"train_rmse => {train_rmse}")
print(f"test_rmse  => {test_rmse}")

# Create a DataFrame with the predicted and original values for ShankAngles and ThighAngles
df = pd.DataFrame({'test_y': test_y, 'pred': pred, 'ShankAngles': test_X['ShankAngles'], 'ThighAngles': test_X['ThighAngles']})

# Create a scatter plot with ShankAngles and ThighAngles on the y-axis, and predicted and original values shown in different colors
plt.scatter(df['pred'], df['ShankAngles'], label='ShankAngles - Predicted', alpha=0.7, marker='o', s=40, color='green', edgecolors='k')
plt.scatter(df['test_y'], df['ShankAngles'], label='ShankAngles - Original', alpha=0.7, marker='o', s=40, color='blue', edgecolors='k')
plt.scatter(df['pred'], df['ThighAngles'], label='ThighAngles - Predicted', alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')
plt.scatter(df['test_y'], df['ThighAngles'], label='ThighAngles - Original', alpha=0.7, marker='o', s=40, color='red', edgecolors='k')
plt.xlabel('Gait Percentage')
plt.ylabel('Angles')
plt.title('Gait Percentage vs. Angles')

# Add RMSE to the plot
plt.text(0.05, 0.95, f"RMSE: {round(rmse, 2)}", transform=plt.gca().transAxes)

plt.legend()
plt.show()
