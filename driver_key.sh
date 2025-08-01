####################
# Input Directories
####################
borzoi_examples_data_dir="/n/groups/price/ben/tools/borzoi/examples/"


####################
# Output Directories
####################
temp_output_root="/n/scratch/users/b/bes710/robust_borzoi/"

borzoi_downloads_dir=${temp_output_root}"borzoi_downloads/"


if false; then
sh download_borzoi_data.sh $borzoi_downloads_dir
fi

sh prelim_experiments.sh $borzoi_downloads_dir $borzoi_examples_data_dir