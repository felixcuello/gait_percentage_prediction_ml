from GPE import xgb_r
import numpy as np
import pandas as pd

# Example input values
new_data = pd.DataFrame({
    'ShankAngles': [123], # Replace 30 with the desired angle value
    'ShankVelocity': [0.5], # Replace 0.5 with the corresponding velocity value
    'ShankAngularVelocity': [np.nan], # Replace 2 with the corresponding angular velocity value
    'ShankIntegral': [np.nan], # This value is not used in prediction and can be set to 0
    'ThighAngles': [np.nan], # This value is not used in prediction and can be set to 0
    'ThighVelocity': [np.nan], # This value is not used in prediction and can be set to 0
    'ThighAngularVelocity': [np.nan], # This value is not used in prediction and can be set to 0
    'ThighIntegral': [np.nan] # This value is not used in prediction and can be set to 0
})


# Predict the gait percentage using the trained XGBoost model
predicted_percentage = xgb_r.predict(new_data)

# Print the predicted gait percentage
print('Predicted gait percentage:', predicted_percentage[0])