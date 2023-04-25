import os
import copy
import logging
import random

from math import isnan

# Setting log level
os_log_level = os.environ['LOG_LEVEL'] if os.environ['LOG_LEVEL'] != None else 'INFO'
logging.root.setLevel(os_log_level)


#  Scale
# ---------------------------------------------
class Scaling:
    def process(self, df, scale_percentage=5, seed=42):
        random.seed(seed)
        affected_columns = [
            'ShankAngles', 'ShankVelocity', 'ShankAngularVelocity', 'ShankIntegral',
            'ThighAngles', 'ThighVelocity', 'ThighAngularVelocity', 'ThighIntegral']
        number_of_rows = len(df)

        # Iterate over all rows
        for row_number in range(0, number_of_rows):
            # add noise to angles
            for col in affected_columns:
                row = copy.copy(df.iloc[row_number])
                if(not isnan(row[col])):
                    row[col] = row[col] * (1 + random.uniform(-scale_percentage, scale_percentage) / 100)
                df.loc[len(df)] = row


        return df

#  Add noise to the Shank and Tigh Angles
# ---------------------------------------------
class NoiseInjection:
    def process(self, df, noise_variability=0.2, seed=42):
        random.seed(seed)
        affected_columns = ['ShankAngles', 'ThighAngles']
        number_of_rows = len(df)

        # Iterate over all rows
        for row_number in range(0, number_of_rows):
            # add noise to angles
            for col in affected_columns:
                row = copy.copy(df.iloc[row_number])
                row[col] = row[col] + random.uniform(0.00, noise_variability)
                df.loc[len(df)] = row

        return df


#  Move percentages up and down
# ---------------------------------------------
class TimeShift:
    def add_shifted_value(self, df, row_number, shift):
        row = copy.copy(df.iloc[row_number])
        row['gait_percentage'] = int(((row['gait_percentage'] + shift) % 100) + 1)
        df.loc[len(df)] = row

    def process(self, df, shift_amount=1):
        number_of_rows = len(df)

        # Iterate over all rows
        for row_number in range(0, number_of_rows):
            # Forward time shift
            # ------------------------------------------------
            if(row_number % 100 == 0):
                logging.debug("Forward TimeShift (rows {}/{})".format(row_number, number_of_rows))

            for shift in range(1, shift_amount+1):
                self.add_shifted_value(df, row_number, shift)

            # Backward time shift
            # ------------------------------------------------
            if(row_number % 100 == 0):
                logging.debug("Backward TimeShift (rows {}/{})".format(row_number, number_of_rows))

            for shift in range(100-shift_amount, 100):
                self.add_shifted_value(df, row_number, shift)

        return df
