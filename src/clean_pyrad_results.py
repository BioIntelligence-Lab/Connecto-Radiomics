'''
@ichaudry
Clean ther results.csv files to only include the "original" features and deal with na values.
'''

import pandas as pd
import sys
from read_parameters import parse_params
import traceback

def clean(rad_dir, num_labels):

    for i in range(num_labels):
        df = pd.read_csv(rad_dir + '/results_pyrad_' + str(i+1) + '.csv')
        clean_df = df[['Image', 'Mask']]

        #think about best way to deal with missing data here
        #####
        clean_df = clean_df.join(df.filter(regex='^original_*').fillna(0))
        #####

        clean_df.to_csv(rad_dir + '/cleaned_results_pyrad_' + str(i+1) + '.csv')

PARAMETERS_FILE = sys.argv[1]
parameter_reader = parse_params(PARAMETERS_FILE)
try:
    clean(parameter_reader['radiomics']['dir'], parameter_reader['segmentation_separation']['num_segmentations'])
except:
    print('Error: Could not clean pyradiomics outputs. See traceback:')
    print(traceback.format_exc())