# land_cover_analysis
Land cover classification and change analysis

# Land Cover classification with machine learning

This code performs land cover classification over Landsat OLI 8 (at Nov 2022)
using GAP (US) and/or INEGI (Mexico) land use and land cover datasets for
training the algorithms.

# Requirements

Code runs on Python 3 using anaconda3 modules, a conda environment with the
following modules should be prepared and activated to run this code (only the
main modules are listed, some additional dependecies may be required for
specific tasks, e.g. visualization):

  * Python 3.8.x
  * numpy 1.20.x
  * matplotlib 3.5.x
  * gdal 3.4.1
  * scikit-learn 1.1.2

**IMPORTANT**: there are known issues between gdal and matplotlib, but the
combination of versions presented above seem to work just fine.

# Environments

The following environments are provided for running the Python scripts on
the **asili** server (asili.ece.arizona.edu):

  * **rs**: environment ready to run various remote sensing scripts, thus
    includes packages such as numpy, gdal, h5py, and matplotlib.
  * **rsml**: environment with the same modules as **rs** but with additional
    machine learning capabilities such as scikit-learn.
  * **gis** pretty much the same modules as **rs**. Created as backup to not
    mess with the main **rs** environment.

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

# Changelog:

  * 2022/11/18: first version of repo structure (actual code created before)
    This file will be inmmediatly updated in local machine at VIPLab, look
    there for a more recent version
