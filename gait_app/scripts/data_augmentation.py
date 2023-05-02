import os
import copy
import logging
import random
import time

from math import isnan

# Setting log level
os_log_level = os.environ['LOG_LEVEL'] if os.environ['LOG_LEVEL'] != None else 'INFO'
logging.root.setLevel(os_log_level)

#  Interpolation
# ---------------------------------------------
#  Creates interstitial
class Interpolation:
    def process(self, df, interpolation_percentage=0.5, seed=42):
        random.seed(seed)
        affected_columns = ['ShankAngles', 'ThighAngles']
        number_of_rows = len(df)

        logging.info("Applying Interpolation (interpolation_percent={}%)".format(round(interpolation_percentage * 100, 4)))
        start_time = time.time()

        df.loc[len(df)] = copy.copy(df.iloc[0]) # Copy the first column
        for row_number in range(0, number_of_rows-1):
            # Get the interpolated values
            row = copy.copy(df.iloc[row_number])
            next_row = copy.copy(df.iloc[row_number+1])
            for col in affected_columns:
                row[col] = (row[col] + next_row[col]) * interpolation_percentage

            df.loc[len(df)] = row

        spent_time = time.time() - start_time
        logging.info("Interpolation took: {:.2f} seconds (result df size={})".format(spent_time, len(df)))

        return df


#  Scale
# ---------------------------------------------
#  Scaling reduces or increases all the values by some proportion
#  It's analogous to zooming
class Scaling:
    def process(self, df, scale_percent=2, seed=42):
        random.seed(seed)
        affected_columns = [
            'ShankAngles', 'ShankVelocity', 'ShankAngularVelocity', 'ShankIntegral',
            'ThighAngles', 'ThighVelocity', 'ThighAngularVelocity', 'ThighIntegral']
        number_of_rows = len(df)

        # Iterate over all rows
        logging.info("Applying Scaling (df size={} / scale_percent={}%)".format(len(df), round(scale_percent, 4)))
        start_time = time.time()
        for row_number in range(0, number_of_rows):
            # add noise to angles
            for col in affected_columns:
                row = copy.copy(df.iloc[row_number])
                if(not isnan(row[col])):
                    row[col] = row[col] * (1 + scale_percent / 100)

            df.loc[len(df)] = row
        spent_time = time.time() - start_time
        logging.info("Scaling took: {:.2f} seconds (result df size={})".format(spent_time, len(df)))

        return df

#  Add noise to the Shank and Tigh Angles
# ---------------------------------------------
#   Noise injection adds a noise to the shank and thigh angle
#   It's analogous to preserve the shape but blurring a bit the data
class NoiseInjection:
    def process(self, df, noise_variability=0.2, seed=42):
        random.seed(seed)
        affected_columns = ['ShankAngles', 'ThighAngles']
        number_of_rows = len(df)

        # Iterate over all rows
        logging.info("Applying NoiseInjection (df size={} / noise_variability={})".format(len(df), noise_variability))
        start_time = time.time()
        for row_number in range(0, number_of_rows):
            # add noise to angles
            for col in affected_columns:
                row = copy.copy(df.iloc[row_number])
                row[col] = row[col] + random.uniform(-noise_variability, noise_variability)

            df.loc[len(df)] = row
        spent_time = time.time() - start_time
        logging.info("Noise injection took: {:.2f} seconds (result df size={})".format(spent_time, len(df)))

        return df


#  Move percentages up and down
# ---------------------------------------------
class TimeShift:
    def add_shifted_value(self, df, row_number, shift):
        row = copy.copy(df.iloc[row_number])
        row['gait_percentage'] = int(((row['gait_percentage'] + shift) % 100) + 1)
        df.loc[len(df)] = row

    def process(self, df, shift_amount=1):
        logging.warn("TimeShift is not working properly, skipping")
        return df

        number_of_rows = len(df)

        # Iterate over all rows
        for row_number in range(0, number_of_rows):
            # Forward time shift
            # ------------------------------------------------
            if(row_number % 100 == 0):
                logging.info("Forward TimeShift (rows {}/{})".format(row_number, number_of_rows))

            for shift in range(1, shift_amount+1):
                self.add_shifted_value(df, row_number, shift)

            # Backward time shift
            # ------------------------------------------------
            if(row_number % 100 == 0):
                logging.info("Backward TimeShift (rows {}/{})".format(row_number, number_of_rows))

            for shift in range(100-shift_amount, 100):
                self.add_shifted_value(df, row_number, shift)

        return df
