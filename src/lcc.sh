#!/usr/bin/bash
# lcc.sh
# Run the land cover classification Python scripts
# Eduardo Jimenez Hernandez <eduardojh@arizona.edu>  May, 2023

version="0.1"
usage="$(basename "$0") [-h] [-v] [MODE SCRIPT] -- run land cover classification scripts

where:
    -h --help      shows this help text and exits
    -v --version   shows script version
    MODE           either: local or server
    SCRIPT         script to run: group sample sampleval fill exploration rf nn all"

if [ "$1" = "-h" ]; then
	echo "$usage"
	exit 0
elif [ "$1" = "--help" ]; then
	echo "$usage"
	exit 0
elif [ "$1" = "-v" ]; then
	echo "$version"
	exit 0
elif [ "$1" = "--version" ]; then
	echo "$version"
	exit 0
elif [ "$1" = "local" ]; then
	echo "Running on Ubuntu Workstation (local)..."
	# Setup custom directories
	PROJ_LIB='/home/eduardo/anaconda3/envs/rsml/share/proj/'
	GDAL_DATA='/home/eduardo/anaconda3/envs/rsml/share/gdal/'
	RS_LIB='/vipdata/2023/land_cover_analysis/lib/'
	DATA_DIR='/vipdata/2023/CALAKMUL/ROI1/'
	
	echo "Initializing..."
	source /home/eduardo/anaconda3/etc/profile.d/conda.sh

	# Activate the anaconda environment
	conda activate rsml

elif [ "$1" = "server" ]; then
	echo "Running on Alma Linux server (remote)..."
	# Setup remote directories
	PROJ_LIB='/home/eduardojh/.conda/envs/rsml/share/proj/'
	GDAL_DATA='/home/eduardojh/.conda/envs/rsml/share/gdal/'
	RS_LIB='/home/eduardojh/Documents/land_cover_analysis/lib/'
	DATA_DIR='/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/ROI1/'
	
	echo "Initializing..."
	# Configure the bash terminal to use the profile
	#source ~/.bash_profile
    #source ~/.bashrc
    source /usr/local/anaconda3/etc/profile.d/conda.sh
	
	# Activate the anaconda environment
	conda activate rsml
    #conda activate /home/eduardojh/.conda/envs/rsml/

else
	echo "Invalid option, choose: local or server. Exiting."
	conda deactivate
	exit 1
fi

echo "Environment configured. Now running main scripts..."
# Load the library first
echo "Loading: $RS_LIB"rsmodule.py
python "$RS_LIB"rsmodule.py $PROJ_LIB $GDAL_DATA $DATA_DIR

if [ "$2" = "group" ]; then
	echo "Running land cover grouping..."
	python 00_group.py $RS_LIB $DATA_DIR
elif [ "$2" = "sample" ]; then
	echo "Running the stratified random sampling to create training/testing datasets..."
	python 01_sample_mask.py $RS_LIB $DATA_DIR
elif [ "$2" = "sampleval" ]; then
	echo "Running the stratified random sampling to create training/validation/testing datasets..."
	python 01_sample_mask_validation.py $RS_LIB $DATA_DIR
elif [ "$2" = "fill" ]; then
	echo "Running dataset preparation and filling..."
	python 02_split_dataset_fill.py $RS_LIB $DATA_DIR
elif [ "$2" = "exploration" ]; then
	echo "Running data exploration..."
	python 03_exploration.py $RS_LIB $DATA_DIR
elif [ "$2" = "rf" ]; then
	echo "Running Random Forest..."
	python 04_landcover_rf.py $RS_LIB $DATA_DIR
elif [ "$2" = "nn" ]; then
	echo "Running Neural Network..."
elif [ "$2" = "all" ]; then
	echo "Running everything! This will take a while..."
	python 00_group.py $RS_LIB $DATA_DIR
	python 01_sample_mask.py $RS_LIB $DATA_DIR
	python 02_split_dataset_fill.py $RS_LIB $DATA_DIR
	python 03_exploration.py $RS_LIB $DATA_DIR
	python 04_landcover_rf.py $RS_LIB $DATA_DIR
else
	echo "Invalid option, choose: group, sample, sampleval, fill, exploration, rf, nn, all. Exiting."
	conda deactivate
	exit 1
fi

conda deactivate

echo "Scripts executed successfully!"
exit 0
