'''
@author ichaudry/ChatGPT

Splitting segmatation predictions into multiple NIFTI files where each NIFTI file is a single segmented element. 

'''

import os
import nibabel as nib
import numpy as np
import sys
from read_parameters import parse_params

def separate_segmentations(input_folder, output_folder, num_segmentations):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all NIFTI files in the input folder
    nifti_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

    for nifti_file in nifti_files:
        # Load the NIFTI file
        img = nib.load(os.path.join(input_folder, nifti_file))
        data = img.get_fdata()

        # Separate each segmentation and save it as a new NIFTI file
        for i in range(num_segmentations):
            # Get a binary mask for the current segmentation label
            segmentation_data = (data == i).astype(np.int16)  # Specify the data type as np.int16

            # Create a new filename based on the original filename and the current label
            base_filename, _ = os.path.splitext(nifti_file)
            base_filename = str(base_filename.split('.')[0])
            new_filename = f'{base_filename}_label_{i+1}.nii.gz'

            # Save the new NIFTI image
            nib.save(nib.Nifti1Image(segmentation_data, img.affine), os.path.join(output_folder, new_filename))
            print(f"Segmentation {i+1} saved to: {new_filename}")




PARAMETERS_FILE = sys.argv[1]
parameter_reader = parse_params(PARAMETERS_FILE)
try:
    separate_segmentations(parameter_reader['segmentation_separation']['dest_dir'], 
                            parameter_reader['segmentation_separation']['segmentation_nifit_seg_path'], 
                            parameter_reader['segmentation_separation']['num_segmentations'])
except:
    print('Error: Could not separate out the segmentations.')

'''
PARAMETERS_FILE = sys.argv[1]

if PARAMETERS_FILE.split('.')[-1] != 'json':
    print('Error: Incorrect parameter file. Requires a JSON formatted as noted in documentation.')
    quit()
else:
    with open(PARAMETERS_FILE, 'r') as f:
        parameter_reader = json.load(f)

        try:
            separate_segmentations(parameter_reader['segmentation_separation']['source_dir'], 
                                   parameter_reader['segmentation_separation']['dest_dir'], 
                                   parameter_reader['segmentation_separation']['num_segmentations'])
        except:
            print('Error: Could not separate out the segmentations.')'''