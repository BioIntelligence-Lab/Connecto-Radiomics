'''
@author ichaudry

a helper function to streamline reading in the parameter file.
'''

import json

def parse_params(file):
    '''
    Reads in the json file and returns the parsed output. 

    Params
    ------
    file: str
        path to the json file
    
    Return
    ------
    params: dict
        the dictionary of the parsed json
    '''


    if file.split('.')[-1] != 'json':
        print('Error: Incorrect parameter file. Requires a JSON formatted as noted in documentation.')
        quit()
    else:
        with open(file, 'r') as f:
            parameter_reader = json.load(f)
            return parameter_reader