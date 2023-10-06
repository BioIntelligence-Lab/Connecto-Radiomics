##############################################################################################################################
# RADCON: main driver script
# AUTHOR: I. Chaudry (ichaudry@som.umaryland.edu)
# A radiomics driven connectiomics pipeline for medical imaging. See README.md for more information. 
##############################################################################################################################

eval "$(conda shell.bash hook)"
conda activate ic_env  ##need to make a conda environemnt for um2ii-radcon (try to install pyradiomics into the same environment)

#paths
src_dir=./src
parameters=$1

#run the segmentation
#MAY NEED TO MODIFY PER USE CASE
####
echo "$(date '+%Y-%m-%d %H:%M:%S') | >Making inferences"
CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i $(jq -r '.segmentation_separation.source_dir' $parameters) -o $(jq -r '.segmentation_separation.dest_dir' $parameters) -t 500 -m 3d_fullres -tr nnUNetTrainerV3_100epochs
####

#generating nifti files for each segmented mask
echo "$(date '+%Y-%m-%d %H:%M:%S') | >Separating niftis for each segmentation"
python $src_dir/sep_segment_nifti.py $parameters

###radiomics
echo "$(date '+%Y-%m-%d %H:%M:%S') | >Starting radiomics"

#generate input files
echo "$(date '+%Y-%m-%d %H:%M:%S') |       >Making radiomics input files"
python $src_dir/make_pyradiomics_input_csvs.py $parameters


#run the pyradiomics in the background for each label
echo "$(date '+%Y-%m-%d %H:%M:%S') |       >Changing CONDA environment"
conda deactivate
conda activate ic_pyrad

echo "$(date '+%Y-%m-%d %H:%M:%S') |       >Parsing parameters"

num_segs=$(jq -r '.segmentation_separation.num_segmentations' $parameters)
pyrad_dir=$(jq -r '.radiomics.dir' $parameters)
pyrad_job_count=$(jq -r '.radiomics.thread_count' $parameters)

echo "$(date '+%Y-%m-%d %H:%M:%S') |       >Running jobs in the background"
for ((label_id = 1; label_id <= $num_segs; label_id++))
do
    pyradiomics $pyrad_dir/pyrad_input_${label_id}.csv -o $pyrad_dir/results_pyrad_${label_id}.csv -f csv --jobs=${pyrad_job_count} &
    job_pids[$label_id]=$!
    echo "$(date '+%Y-%m-%d %H:%M:%S') |             >Started radiomics for label $label_id"
done

for ((label_id = 1; label_id <= $num_segs; label_id++))
do
    wait "${job_pids[$label_id]}"
done

conda deactivate
conda activate ic_env

echo "$(date '+%Y-%m-%d %H:%M:%S') | >Radiomics done! Cleaning the radiomics results now."
python $src_dir/clean_pyrad_results.py $parameters

#connectomics
echo "$(date '+%Y-%m-%d %H:%M:%S') | >Starting connectomics"
python $src_dir/connectomics.py $parameters
echo "$(date '+%Y-%m-%d %H:%M:%S') | >Finished connectomics"
