'''
makes the csv files to put input in to pyradiomics.

csv is organized with 2 cols: [image_path] and [mask_path]
the mask path is going to be specific to one of the 6 labels
'''

import sys
import csv
import os
import traceback

from read_parameters import parse_params

def make_pyrad_inputs(path_to_images, path_to_labels, num_segmentations, output_dir):
    labels = [i+1 for i in range(num_segmentations)]

    for label in labels:

        rows_to_write = [['Image' , 'Mask']]

        label_file = path_to_labels + '/{image}_label_{label}.nii.gz'

        for file in os.listdir(path_to_images):

            #this may need to be adjusted based off what model is being used to infer segmentations
            #####
            base_file = str(file.split('_0000')[0])
            #####
            
            rows_to_write.append([str(path_to_images + '/' + file), label_file.format(image=base_file, label=label)])

            print(file)

        with open(output_dir+'/pyrad_input_' + str(label) + '.csv', 'w+') as ofile:
            writer = csv.writer(ofile)
            writer.writerows(rows_to_write)


PARAMETERS_FILE = sys.argv[1]
parameter_reader = parse_params(PARAMETERS_FILE)
try:
    make_pyrad_inputs(parameter_reader['segmentation_separation']['source_dir'], 
                            parameter_reader['segmentation_separation']['segmentation_nifit_seg_path'], 
                            parameter_reader['segmentation_separation']['num_segmentations'],
                            parameter_reader['radiomics']['dir'])
except:
    print('Error: Could not make pyradiomics inputs. See traceback:')
    print(traceback.format_exc())