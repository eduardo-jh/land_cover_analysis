# land_cover_analysis
Land cover classification and change analysis

![](data/river.png)

# Land Cover classification with machine learning

This code performs land cover classification over Landsat OLI 8 (at Nov 2022)
using GAP (US) and/or INEGI (Mexico) land use and land cover datasets for
training the algorithms.

# Requirements

Code runs on Python 3 using anaconda3 modules, a conda environment with the
following modules should be prepared and activated to run this code (only the
main modules are listed, some additional dependecies may be required for
specific tasks, e.g. visualization):

  * python 3.8.x
  * numpy 1.23.x
  * matplotlib 3.6.x
  * gdal 3.5.x
  * scikit-learn 1.1.2
  * tensorflow 2.9.x
  * keras 2.9.x

**IMPORTANT**: there are known issues between **gdal** and **matplotlib**, but the
combination of versions presented above seem to work just fine.

# Environments

The following environments are provided for running the Python scripts on
the **asili** server (asili.ece.arizona.edu):

  * **rs**: environment ready to run various remote sensing scripts, thus
    includes packages such as numpy, gdal, h5py, and matplotlib.
  * **rsml**: environment with the same modules as **rs** but with additional
    machine learning capabilities such as scikit-learn.
  * **rsft** pretty much the same modules as **rs** but with additional 
    keras and tensorflow capabilities for neural networks.

To activate any of these environments use:

```
$ conda activate rs
```

or

```
$ conda activate rsml
```

Look for the comments at the first lines of each script for instructions
on which environment to activate before running it.

# Data preprocessing

Preprocessing tasks are carried out with specific tools created in the VIP lab
by Dr. Armando Barreto in the format of Linux binaries. They take as input
Landsat 8 OLI scenes (GeoTIFF) located at specific directories and create
HDF4 files as output.
All the parameters for these programs should be provided in the
terminal or through Bash scripts. Look in the **asili** home directory for
these scripts and instructions.

There are three main 

  1. **Mosaicking**. Stiches Landsat scenes from the input and extracts a
    mosaic for the region of interest in HDF4 format. Options to specify
    the bands are provided.
  2. **Statistics**. Creates files with mean, max, min, and standard
    deviation of each band in the Landsat scenes. Results are presented as
    yearly and/or monthly summaries.
  3. **Phenology**. Calculates phenology metrics based on vegetation indices,
    such as NDVI.

These files are utilized as features/predictors of the machine learning
algorithms and used for training. The land cover classes from GAP or INEGI
are used as labels.

# Modules

The package is composed of a main module and several scripts:

  * **rsmodule.py**: main module with functions for remote sensing
    calculations, plotting, and other utilities.
  * **00_group.py**: groups the land cover classes and reclasiffies raster
    files accordingly, this is useful to merge classes with similar spectral
    signature.
  * **01_sample_mask.py**: uses stratified random sampling to create a mask that
    will be used to split the data into training and testing datasets.
  * **01_sample_mask_validation.py**: uses stratified random sampling to create
    masks that will later be used to split the data into training, validation,
    and testing datasets. This is useful for neural networks.
  * **02_dataset.py**: creates a HDF5 file with the spectral bands, vegetation
    indices, and phenology metrics. It also splits the data into trainig and
    testing datasets according to the mask created previously.
  * **02_split_dataset_fill.py** creates HDF file with spectral bands, VIs,
    and phenology metrics. Splits into training and testing datasets. Also
    fills the NoData (NaN) values.
  * **03_exploration.py**: data exploration plots (histograms, etc.).
  * **04_landcover_rf.py**: land cover classification using random forest.
  * **04_landcover_nn.py** land cover classification using neural netwotks.
  

# Running

To run the script please use the bash script named **lcc.sh** located in
the **src/** directory. Use **--help** to see usage.

# Changelog:

  * 2022/11/18: first version of repo structure (actual code created before)
    This file will be inmmediatly updated in local machine at VIPLab, look
    there for a more recent version.
  * 2023/05/19: completed the main module, scripts, and main bash script.
