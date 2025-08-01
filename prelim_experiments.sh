#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-9:30                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=10GB                         # Memory total in MiB (for all cores)


borzoi_downloads_dir="${1}"
borzoi_examples_data_dir="${2}"


module load gcc/11.5.0 conda/miniforge3
conda activate borzoi_py310


python prelim_experiments.py $borzoi_downloads_dir $borzoi_examples_data_dir