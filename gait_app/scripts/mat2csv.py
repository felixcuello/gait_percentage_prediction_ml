# Libraries
import os
import re
import csv
import sys
import logging
from scipy.io import loadmat

# Setting log level
os_log_level = os.environ['LOG_LEVEL'] if os.environ['LOG_LEVEL'] != None else 'INFO'
logging.root.setLevel(os_log_level)

# Checking parameters
if len(sys.argv) != 3:
    print("Usage {} <mat_file> <csv_file>".format(sys.argv[0]))
    sys.exit()

# Setting the matlab variables
mat_file = sys.argv[1]
mat_data = loadmat(mat_file)

# Checking matlab columns
for col in mat_data.keys():
    # Only process the non administrative column (non __)
    if not re.search("__", col):
        csv_filename = sys.argv[2]
        with open(csv_filename, 'w', newline='') as csv_file:
            logging.info('Processing {} converting into {}'.format(mat_file, csv_filename))
            writer = csv.writer(csv_file)
            subjects = []
            for subject_id in range(0,21):
                subjects.append("subject_{}".format(subject_id))
            writer.writerow(subjects)
            for row in mat_data[col]:
                writer.writerow(row)
