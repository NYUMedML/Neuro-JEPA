#!/bin/bash
#SBATCH --job-name=register_t1w
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu_short
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load miniconda3
module load fsl/6.0.7
module load freesurfer/7.4.1
module load ants

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export MATLAB_HOME=/path/to/matlab/bin
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab
export SPM_HOME=/path/to/spm12
export FSLDIR=/path/to/fsl/6.0.7

# Define input CSV file
CSV_FILE="/path/to/input/data/T1w_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_DIR="/path/to/output/BIDS_registered"
template_img="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"

mkdir -p "$OUTPUT_DIR"

# Read CSV file line by line, skipping the header
tail -n +2 "$CSV_FILE" | while IFS=, read -r file_path; do
    sub_id=$(echo "$file_path" | sed -n 's#.*/sub-\([^/]*\)/.*#\1#p')
    ses_id=$(echo "$file_path" | sed -n 's#.*/ses-\([^/]*\)/anat.*#\1#p')
    out_dir="${OUTPUT_DIR}/sub-${sub_id}/ses-${ses_id}/anat"
    
    base=$(basename "$file_path" .nii.gz)
    mkdir -p "$out_dir"
    echo "Processing: $file_path"

    fslreorient2std "$file_path" "${out_dir}/${base}_reoriented.nii.gz"
    robustfov -i "${out_dir}/${base}_reoriented.nii.gz" -r "${out_dir}/${base}_acpc.nii.gz"
    N4BiasFieldCorrection -d 3 -i "${out_dir}/${base}_acpc.nii.gz" \
          -o "${out_dir}/${base}_n4.nii.gz" || continue
    mri_synthstrip -i "${out_dir}/${base}_n4.nii.gz" -o "${out_dir}/${base}_brain.nii.gz" || continue
    flirt -in "${out_dir}/${base}_brain.nii.gz" -ref "$template_img" \
          -omat "${out_dir}/${base}_affine.mat" -dof 12 -cost corratio
    flirt -interp spline -in "${out_dir}/${base}_brain.nii.gz" -ref "$template_img" \
          -applyxfm -init "${out_dir}/${base}_affine.mat" -out "${out_dir}/${base}.nii.gz"

    rm -f "${out_dir}/${base}_reoriented.nii.gz" "${out_dir}/${base}_acpc.nii.gz" \
          "${out_dir}/${base}_n4.nii.gz" "${out_dir}/${base}_brain.nii.gz" \
          "${out_dir}/${base}_affine.mat"
    echo "Completed: $base -> $out_dir"
done

echo "All subjects processed successfully."
