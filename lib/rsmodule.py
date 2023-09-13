#!/usr/bin/env python
# coding: utf-8

""" NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.4.1 & Matplotlib 3.5.2 from conda-forge)

Some remote sensing and GIS utilities

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2022-07-12

Changelog:
    Jul 12, 2022: Quality assessment of 'pixel_qa', 'sr_aerosol' and 'radsat_qa' bands
    Aug 15, 2022: Creation of MODIS-like QA using Landsat's 'pixel_qa' and 'sr_aerosol'
    Jan 13, 2023: Functions to prepare raster datset for training machine learning (functions from 'San Juan River' script)
    Jan 17, 2023: Land cover percentage analysis on training rasters and new format in function definitions
    Sep 10, 2023: Some functions updated, classes to train a single RF per tile (later updated in OOP)
"""
import sys
import gc
import os

if len(sys.argv) == 4:
    # Check if arguments were passed from terminal
    args = sys.argv[1:]
    os.environ['PROJ_LIB'] = args[0]
    os.environ['GDAL_DATA'] = args[1]
    cwd = args[2]
    print(f"  Using PROJ_LIB={args[0]}")
    print(f"  Using GDAL_DATA={args[1]}")
    print(f"  Using CWD={args[2]}")
else:
    import platform
    system = platform.system()
    if system == 'Windows':
        # On Windows 10
        os.environ['PROJ_LIB'] = 'D:/anaconda3/envs/rsml/Library/share/proj'
        os.environ['GDAL_DATA'] = 'D:/anaconda3/envs/rsml/Library/share'
        cwd = 'D:/Desktop/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/vipdata/2023/CALAKMUL/ROI1/'):
        # On Ubuntu Workstation
        os.environ['PROJ_LIB'] = '/home/eduardo/anaconda3/envs/rsml/share/proj/'
        os.environ['GDAL_DATA'] = '/home/eduardo/anaconda3/envs/rsml/share/gdal/'
        cwd = '/vipdata/2023/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/'):
        # On Alma Linux Server
        os.environ['PROJ_LIB'] = '/home/eduardojh/.conda/envs/rsml/share/proj/'
        os.environ['GDAL_DATA'] = '/home/eduardojh/.conda/envs/rsml/share/gdal/'
        cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/'
    else:
        print('  System not yet configured!')

import csv
import random
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Dict
from pyhdf.SD import SD, SDC
from matplotlib import colors
from osgeo import gdal
from osgeo import osr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.style.use('ggplot')  # R-like plots

# Load feature valid ranges from file
ranges = pd.read_csv(cwd + 'parameters/valid_ranges', sep='=', index_col=0)
MIN_BAND = ranges.loc['MIN_BAND', 'VALUE']
MAX_BAND = ranges.loc['MAX_BAND', 'VALUE']
MIN_VI = ranges.loc['MIN_VI', 'VALUE']
MAX_VI = ranges.loc['MAX_VI', 'VALUE']
MIN_PHEN = ranges.loc['MIN_PHEN', 'VALUE']
NAN_VALUE = ranges.loc['NAN_VALUE', 'VALUE']


class LandCoverDataset:

    NA_CLASS = 0  # In raster 0=NoData, other values are land cover classes
    # NO_DATA = 0  # np.nan

    def __init__(self, datadir: str, phendir: str, cwd: str, fn_landcover: str, phenobased="NDVI") -> None:
        """ Initializes objects to proper directories

        LABELS: one single GeoTIFF raster file
        FEATURES: multi-dimension HDF4 tiled files

        Assummes everything is projected in the same spatial reference as the file in the parameters directory

        :param datadir: main data directory with surface reflectances and VIs in HDF4
        :param phendir: phenology data in HDF4
        :param cwd: directory to locate labels and to write in HDF5 features
        """

        print("Initializing land cover classification")
        self.datadir = datadir
        self.phendir = phendir
        self.cwd = cwd
        self.phenobased = phenobased  # either NDVI or EVI (or EVI2)
        assert os.path.isdir(self.datadir), f"Directory doesn't exist: {self.datadir}"
        assert os.path.isdir(self.phendir), f"Directory doesn't exist: {self.phendir}"
        assert os.path.isdir(self.cwd), f"Directory doesn't exist: {self.cwd}"
        print(f"Data directory: {self.datadir}")
        print(f"Phenology directory: {self.datadir}")
        print(f"Current working directory: {self.cwd}")

        # Setup the files required to split the label raster into tiles
        self.fn_spatial_ref = os.path.join(cwd, 'parameters/spatial_reference')  # custom spatial reference
        self.fn_tiles = os.path.join(cwd, 'parameters/tiles')  # extent of each tile

        # Setup file names for sampling section
        self.fn_keys = os.path.join(cwd, 'data/inegi_2018/land_cover_groups.csv')
        self.fn_pixelcount_plot = os.path.join(cwd, 'sampling/ROI2_percent_plot.png')
        self.fn_training_mask = os.path.join(cwd, 'sampling/ROI2_training_mask.tif')
        self.fn_training_labels = os.path.join(cwd, 'sampling/ROI2_training_labels.tif')
        self.fn_sample_sizes = os.path.join(cwd, 'sampling/dataset_sample_sizes.csv')

        # Setup file names for creation of feature datasets
        # self.fn_phenology = None
        # self.fn_phenology2 = None
        # Create files to save features and parameters
        # self.fn_features = None
        self.fn_labels = None
        self.fn_parameters = None
        self.fn_feat_indices = None

        # Initialize variables
        self.landcover = None
        self.nodata = None
        self.epsg = None
        self.metadata = None
        self.geotransform = None
        self.projection = None
        self.spatial_reference = None
        self.ncols = None
        self.nrows = None

        self.train_percent = 0.2
        self.window_size = 7
        self.max_trials = int(2e5)
        self.training_mask = None
        self.training_labels = None

        self.fill = False
        self.normalize = False
        self.standardize = False
        self.no_data_arr = None

        self.tiles = None
        self.tiles_extent = None
        self.tile_rows = 5000
        self.tile_cols = 5000

        self.feat_indices = None
        self.feat_names = None
        self.feat_indices_season = None
        self.feat_names_season = None

        self.bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
        self.band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
        self.months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        self.vars = ['AVG', 'STDEV']
        self.phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
        self.phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']
        
        # Open land cover labels
        self.fn_landcover = self.cwd + fn_landcover
        self.read_land_cover()
        self.read_spatial_reference()


    def read_spatial_reference(self):
        with open(self.fn_spatial_ref, 'r') as f:
            self.spatial_reference = f.read()
        print(f"Setting spatial reference: {self.spatial_reference}")
    

    def read_land_cover(self):
        assert os.path.isfile(self.fn_landcover) is True, f"ERROR: File not found! {self.fn_landcover}"
        
        self.landcover, self.nodata, self.metadata, self.geotransform, self.projection, self.epsg = open_raster(self.fn_landcover)
        self.landcover = self.landcover.astype(np.int8)  # convert to byte (int8)
        self.landcover = self.landcover.filled(0) # replace masked constant "--" with zeros
        self.epsg = np.uint16(self.epsg)  # uint16 == ushort
        self.nodata = np.int8(self.nodata)

        self.nrows = self.landcover.shape[0]
        self.ncols = self.landcover.shape[1]
        
        print(f'  -- Opening raster : {self.fn_landcover}')
        print(f'  ---- Metadata     : {self.metadata}')
        print(f'  ---- NoData       : {self.nodata}')
        print(f'  ---- Columns      : {self.ncols}')
        print(f'  ---- Rows         : {self.nrows}')
        print(f'  ---- Geotransform : {self.geotransform}')
        print(f'  ---- Projection   : {self.projection}')
        print(f'  ---- EPSG         : {self.epsg}')
        print(f'  ---- Type         : {self.landcover.dtype}')


    def set_spatial_reference(self, spatial_reference):
        """ Sets custom spatial reference """
        self.spatial_reference = spatial_reference
        print(f"Setting spatial reference: {self.spatial_reference}")
    

    def get_spatial_reference(self):
        return self.spatial_reference
    

    def set_tiles(self, tile_list):
        self.tiles =  tile_list


    def get_tiles(self):
        return self.tiles
    

    def incorporate_ancillary(self, ancillarydir: str, ancillary_files: dict) -> None:
        """ Incorporates a list of ancillary rasters to its corresponding land cover class
        Ancillary data has to be 1's for data and 0's for NoData
        """

        for key in ancillary_files.keys():
            for ancillary_file in ancillary_files[key]:
                fn_ancillary = os.path.join(ancillarydir, ancillary_file)
                ancillary, _, _, _, _, _ = open_raster(fn_ancillary)
                assert ancillary.shape == landcover.shape, f"{ancillary_file} and landcover dimensions don't match"
                landcover = np.where(ancillary > 0, key, landcover)
        
        # Save the land cover with the integrated ancillary data
        self.fn_landcover = self.landcover[:-4] + "_ancillary.tif"
        create_raster_proj(self.fn_landcover, landcover, self.spatial_reference, self.geotransform)
        print(f"Land cover file is now: {self.fn_landcover}")
        self.read_land_cover()  # update land cover


    def split_raster(self, fn_input: str, tile_names: List, fn_tiles_ext: str, outdir: str, custom_proj4: str) -> None:
        """ Split a raster into a mosaic of smaller rasters """

        print("\nSplitting raster into tiles.")
        
        # Open raster
        landcover, nodata, metadata, geotransform, projection, epsg = open_raster(fn_input)
        print(f'  Opening raster  : {fn_input}')
        print(f'  -- Metadata     : {metadata}')
        print(f'  -- NoData       : {nodata}')
        print(f'  -- Columns      : {landcover.shape[1]}')
        print(f'  -- Rows         : {landcover.shape[0]}')
        print(f'  -- Geotransform : {geotransform}')
        print(f'  -- Projection   : {projection}')
        print(f'  -- EPSG         : {epsg}')
        print(f'  -- Type         : {landcover.dtype}')

        nrows, ncols = landcover.shape[0], landcover.shape[1]

        # Extent will be N-S, and W-E boundaries
        merged_ext = {}
        merged_ext['W'], xres, _, merged_ext['N'], _, yres = [int(x) for x in geotransform]
        merged_ext['E'] = merged_ext['W'] + ncols*xres
        merged_ext['S'] = merged_ext['N'] + nrows*yres

        print(merged_ext)

        # Put the extent of tiles into a dictionary
        tiles_extent = {}
        with open(fn_tiles_ext, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # print(row)
                row_dict = {}
                for item in row[1:]:
                    itemlst = item.split('=')
                    row_dict[itemlst[0].strip()] = int(float(itemlst[1]))
                tiles_extent[row[0]] = row_dict
        # print(tiles_extent)

        for tile in tile_names:
            newpath = os.path.join(outdir, tile)
            if not os.path.exists(newpath):
                print(f"\nCreating new path: {newpath}")
                os.makedirs(newpath)
            
            # Calculate coodinates of tile to extract
            tile_ext = tiles_extent[tile]

            nrow = (tile_ext['N'] - merged_ext['N'])//yres
            srow = (tile_ext['S'] - merged_ext['N'])//yres
            wcol = (tile_ext['W'] - merged_ext['W'])//xres
            ecol = (tile_ext['E'] - merged_ext['W'])//xres
            print(f"{tile}: N={nrow} S={srow} W={wcol} E={ecol}")
            
            tile_geotransform = (tile_ext['W'], xres, 0, tile_ext['N'], 0, yres)
            print(f"Geotransform: {tile_geotransform}")

            # Slice the data from raster
            tile_landcover = landcover[nrow:srow, wcol:ecol]
            print(f"Slice: {nrow}:{srow+1}, {wcol}:{ecol+1} {tile_landcover.shape}")

            # Save the sliced data into a new raster
            fn_tile = os.path.join(newpath, f"usv250s7cw2018_ROI2_{tile}.tif")
            print(fn_tile)
            create_raster_proj(fn_tile, tile_landcover, custom_proj4, tile_geotransform)


    def sample(self, training_percent=0.2, win_size=7, maximum_trials=2e5):
        """ Stratified random sampling

        :param float train_percent: default training-testing proportion is 80-20%
        :param int win_size: default is sampling a window of 7x7 pixels
        :param int max_trials: max of attempts to fill the sample size
        """

        self.train_percent = training_percent
        self.window_size = win_size
        self.max_trials = int(maximum_trials)       

        # Create a list of land cover keys and its area covered percentage
        # assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
        lc_frq = land_cover_freq(self.fn_landcover, self.fn_keys, verbose=False)
        print(f'  --Land cover freqencies: {lc_frq}')

        lc_lbl = list(lc_frq.keys())
        freqs = [lc_frq[x] for x in lc_lbl]  # pixel count
        percentages = (freqs/sum(freqs))*100  # percent, based on pixel count

        # Plot land cover percentage horizontal bar
        print('  --Plotting land cover percentages...')
        plot_land_cover_hbar(lc_lbl, percentages, self.fn_pixelcount_plot,
            title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
            xlabel='Percentage (based on pixel count)',
            ylabel='Land Cover (Grouped)',  # remove if not grouped
            xlims=(0,100))

        #### Sample size == testing dataset
        # Use a dataframe to calculate sample size
        df = pd.DataFrame({'Key': lc_lbl, 'PixelCount': freqs, 'Percent': percentages})
        df['TrainPixels'] = (df['PixelCount']*self.train_percent).astype(int)
        # print(df['TrainPixels'])

        # # Undersample largest classes to compensate for unbalance
        # max_val = df['TrainPixels'].max()
        # fix_val = (df.loc[df['TrainPixels'] == max_val, 'PixelCount']*(1-test_percent)).astype(int)
        # df.loc[df['TrainPixels'] == max_val, 'TrainPixels'] = fix_val

        # Now calculate percentages
        df['TrainPercent'] = (df['TrainPixels'] / df['PixelCount'])*100
        print(df)

        # #### 2. Create the testing mask
        # # assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
        # raster_arr, nd, meta, gt, proj, epsg = rs.open_raster(fn_landcover)
        # print(f'  --Opening raster : {fn_landcover}')
        # print(f'  ----Metadata     : {meta}')
        # print(f'  ----NoData       : {nd}')
        # print(f'  ----Columns      : {raster_arr.shape[1]}')
        # print(f'  ----Rows         : {raster_arr.shape[0]}')
        # print(f'  ----Geotransform : {gt}')
        # print(f'  ----Projection   : {proj}')
        # print(f'  ----EPSG         : {epsg}')
        # print(f'  ----Type         : {raster_arr.dtype}')

        nrows, ncols = self.landcover.shape
        print(f"  --Total pixels={nrows*ncols}, Values={sum(df['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df['PixelCount'])}")

        # raster_arr = raster_arr.astype(int)
        # print(f"  --Before filling NoData: {np.unique(self.landcover)}")
        # self.landcover = self.landcover.filled(0)  # replace masked constant "--" with zeros
        # print(f"  --After filling NoData: {np.unique(raster_arr)}")
        # print(f'  --Check new array type: {raster_arr.dtype}')

        sample = {}  # to save the sample

        # Create a mask of the sampled regions
        self.training_mask = np.zeros(self.landcover.shape, dtype=self.landcover.dtype)

        # A window will be used for sampling, this array will hold the sample
        window_sample = np.zeros((self.window_size,self.window_size), dtype=int)

        print(f'  --Max trials: {self.max_trials}')

        trials = 0  # attempts to complete the sample
        completed = {}  # classes which sample is complete

        for sample_key in list(df['Key']):
            completed[sample_key] = False
        completed_samples = sum(list(completed.values()))  # Values are all True if completed
        total_classes = len(completed.keys())
        # print(completed)

        sampled_points = []

        while (trials < self.max_trials and completed_samples < total_classes):
            show_progress = (trials%10000 == 0)  # Step to show progress
            if show_progress:
                print(f'  --Trial {1 if trials == 0 else trials:>8} of {self.max_trials:>8} ', end='')

            # 1) Generate a random point (row_sample, col_sample) to sample the array
            #    Coordinates relative to array positions [0:nrows, 0:ncols]
            #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
            col_sample = random.randrange(0 + self.window_size//2, ncols - self.window_size//2, self.window_size)
            row_sample = random.randrange(0 + self.window_size//2, nrows - self.window_size//2, self.window_size)

            # Save the points previously sampled to avoid repeating and oversampling
            point = (row_sample, col_sample)
            if point in sampled_points:
                trials +=1
                continue
            else:
                sampled_points.append(point)

            # 2) Generate a sample window around the random point, here create the boundaries,
            #    these rows and columns will be used to slice the sample
            win_col_ini = col_sample - self.window_size//2
            win_col_end = col_sample + self.window_size//2 + 1  # add 1 to slice correctly
            win_row_ini = row_sample - self.window_size//2
            win_row_end = row_sample + self.window_size//2 + 1

            assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
            assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

            # 3) Check if sample window is out of range, if so trim the window to the array's edges accordingly
            #    This may not be necessary if half the window size is subtracted, but still
            if win_col_ini < 0:
                # print(f'    --Adjusting win_col_ini: {win_col_ini} to 0')
                win_col_ini = 0
            if win_col_end > ncols:
                # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
                win_col_end = ncols
            if win_row_ini < 0:
                # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
                win_row_ini = 0
            if  win_row_end > nrows:
                # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
                win_row_end = nrows

            # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
            window_sample = self.landcover[win_row_ini:win_row_end,win_col_ini:win_col_end]
            # print(window_sample)
            
            # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
            sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
            classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

            # 6) Iterate over each class sample and add its respective pixel count to the sample
            for sample_class, class_count in zip(sample_keys, sample_freq):
                if sample_class == self.NA_CLASS:
                    # Sample is mixed with zeros, tag it to remove it and go to next sample_class
                    classes_to_remove.append(sample_class)
                    continue

                if completed.get(sample_class, False):
                    classes_to_remove.append(sample_class)  # do not add completed classes to the sample
                    continue

                # Accumulate the pixel counts, chek first if general sample is completed
                if sample.get(sample_class) is None:
                    sample[sample_class] = class_count
                else:
                    # if sample[sample_class] < sample_sizes[sample_class]:
                    sample_size = df[df['Key'] == sample_class]['TrainPixels'].item()

                    # If sample isn't completed, add the sampled window
                    if sample[sample_class] < sample_size:
                        sample[sample_class] += class_count
                        # Check if last addition completed the sample
                        if sample[sample_class] >= sample_size:
                            completed[sample_class] = True  # this class' sample is now complete
                            # but do not add to classes_to_remove
                    else:
                        # This class' sample was completed already
                        completed[sample_class] = True
                        classes_to_remove.append(sample_class)

            # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
            sampled_window = np.zeros(window_sample.shape, dtype=self.landcover.dtype)
            
            # Filter out classes with already complete samples
            if len(classes_to_remove) > 0:
                for single_class in classes_to_remove:
                    # Put a 1 on a complete class
                    filter_out = np.where(window_sample == single_class, 1, 0)
                    sampled_window += filter_out
                
                # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
                sampled_window = np.where(sampled_window == 0, 1, 0)
            else:
                sampled_window = window_sample[:,:]
            
            # Slice and insert sampled window
            self.training_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window

            trials += 1

            completed_samples = sum(list(completed.values()))  # Values are all True if completed
            if show_progress:
                print(f' (completed {completed_samples:>2}/{total_classes:>2} samples)')
            if completed_samples >= total_classes:
                print(f'\n  --All samples completed in {trials} trials! Exiting.\n')

        if trials == self.max_trials:
            print('\n  --WARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

        print(f'  --Sample sizes per class: {sample}')
        print(f'  --Completed samples: {completed}')

        print('\n  --WARNING! This may contain oversampling caused by overlapping windows!')
        df['SampledPixels'] = [sample.get(x,0) for x in df['Key']]
        df['SampledPercent'] = (df['SampledPixels'] / df['TrainPixels']) * 100
        df['SampledPerClass'] = (df['SampledPixels'] / df['PixelCount']) * 100
        df['SampleComplete'] = [completed[x] for x in df['Key']]
        df.to_csv(self.fn_sample_sizes)
        print(df)

        # Convert the training_mask to 1's (indicating pixels to sample) and 0's
        self.training_mask = np.where(self.training_mask >= 1, 1, 0)
        print(f"  --Values in mask: {np.unique(self.training_mask)}")  # should be 1 and 0

        # # To undersample, flip the training/testing pixels of the biggest class
        # # flip_mask = np.where((raster_arr == 3) & (training_mask == 1), 0, training_mask)
        # # training_mask = np.where((raster_arr == 3) & (training_mask == 0), 1, flip_mask)

        # Create a raster with actual labels (land cover classes)
        self.training_labels = np.where(self.training_mask > 0, self.landcover, 0)
        # compl_labels = np.where(training_mask == 0, raster_arr, 0)
        # create_raster(self.fn_training_labels, training_labels, self.epsg, self.geotransform)
        create_raster_proj(self.fn_training_labels, self.training_labels, self.spatial_reference, self.geotransform)

        # Create a raster with the sampled windows, this will be the sampling mask
        # create_raster(self.fn_training_mask, training_mask, self.epsg, self.geotransform)
        create_raster_proj(self.fn_training_mask, self.training_mask, self.spatial_reference, self.geotransform)


    def read_training_mask(self):
        """ Opens raster sample mask, doesn't change other parameters """
        # WARNING! Assumes spatial reference is same as object's default!
        self.training_mask, _, _, _, _, _ = open_raster(self.fn_training_mask)
        self.training_mask = self.training_mask.astype(self.landcover.dtype)


    def read_training_labels(self):
        """ Opens raster with sample labels, doesn't change other parameters """
        # WARNING! Assumes spatial reference is same as object's default!
        self.training_labels, _, _, _, _, _  = open_raster(self.fn_training_labels)
        self.training_labels = self.training_labels.astype(self.landcover.dtype)


    def generate_feature_list(self, fn_feat_list: str='', **kwargs):

        if fn_feat_list != '':
            if os.path.isfile(fn_feat_list):
                # Use a list of selected features instead of generating them as above
                print(f'Trying to get features from: {fn_feat_list}')
                self.feat_indices = []
                self.feat_names = []
                feature = 0
                content = ""
                with open(fn_feat_list, 'r') as f:
                    content = f.readlines()
                for i, line in enumerate(content):
                    line = line.strip()
                    if line != "":
                        self.feat_names.append(line)
                        self.feat_indices.append(feature)
                        print(f"{feature}: {line}")
                        feature += 1
            else:
                print(f"\nERROR: Failed to read features from {fn_feat_list}! Generating features instead.")
        else:
            # All features actually used for classification
            # bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
            # band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']

            # Select appropriate bands according to VI's phenology-based
            if self.phenobased == "NDVI":
                bands = ['Blue', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
                band_num = ['B2', 'B3', 'B7', '', 'B5', 'B4', 'B6']
            elif self.phenobased == "EVI":
                bands = ['Blue', 'Evi', 'Green', 'Mir', 'Nir', 'Red', 'Swir1']
                band_num = ['B2', '', 'B3', 'B7', 'B5', 'B4', 'B6']
            elif self.phenobased == "EVI2":
                bands = ['Blue', 'Evi2', 'Green', 'Mir', 'Nir', 'Red', 'Swir1']
                band_num = ['B2', '', 'B3', 'B7', 'B5', 'B4', 'B6']
            
            months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            # nmonths = [x for x in range(1, 13)]
            vars = ['AVG', 'STDEV']
            phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
            phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

            # # Test a small subset for classification
            # bands = ['Blue', 'Evi', 'Ndvi', 'Nir', 'Red', 'Swir1']
            # band_num = ['B2', '', '', 'B5', 'B4', 'B6']
            # months = ['JAN', 'MAR', 'APR', 'JUN', 'JUL', 'SEP', 'NOV', 'DEC']
            # # nmonths = [x for x in range(1, 13)]
            # nmonths = [1,3,4,6,7,9,11,12]
            # vars = ['AVG']
            # phen = ['SOS', 'EOS', 'DOP', 'MAX', 'NOS']
            # phen2 = ['SOS2', 'EOS2', 'DOP2', 'CUM']

            # Generate feature names...
            print('Generating feature names from combination of variables.')
            self.feat_indices = []
            self.feat_names = []
            feature = 0
            for j, band in enumerate(bands):
                for i, month in enumerate(months):
                    for var in vars:
                        # Create the name of the dataset in the HDF
                        feat_name = month + ' ' + band_num[j] + ' (' + band + ') ' + var
                        if band_num[j] == '':
                            feat_name = month + ' ' + band.upper() + ' ' + var
                        # print(f'  Feature: {feature} Variable: {feat_name}')
                        self.feat_names.append(feat_name)
                        self.feat_indices.append(feature)
                        feature += 1
            for param in phen+phen2:
                feat_name = 'PHEN ' + param
                # print(f'  Feature: {feature} Variable: {feat_name}')
                self.feat_names.append(feat_name)
                self.feat_indices.append(feature)
                feature += 1


    def generate_tile_features(self, tile, nodata_filter, fill=False, normalize=False, standardize=False):

        self.fill = fill
        self.standardize = standardize
        self.normalize = normalize

        self.generate_feature_list()
        # self.generate_feature_list(self.cwd + 'data_exploration/feat_anal/variables.txt')  # from file
        
        # Either normalize or standardize, not both! Normalize has priority
        if self.normalize and self.standardize:
            print("Normalize and standardize both True, setting standardize to False")
            self.standardize = False

        # Use tile filter to only extract pixels valid data
        # assert self.no_data_arr is not None, "ERROR: 'No Data' array should be generated first!"
        assert self.tile_rows == nodata_filter.shape[0], "ERROR: 'No Data' array dimensions do not match!"
        assert self.tile_cols == nodata_filter.shape[1], "ERROR: 'No Data' array dimensions do not match!"
        
        # tile_features = np.zeros((self.tile_rows, self.tile_cols), dtype=np.int)
        # tile_features = np.empty((self.tile_rows, self.tile_cols, len(self.feat_indices)), dtype=np.int16)
        assert len(self.feat_indices) == len(self.feat_names), "Feature name and indices should have same length"
        tile_features = np.zeros((self.tile_rows, self.tile_cols, len(self.feat_indices)), dtype=np.int16)
        
        for n, feat in zip(self.feat_indices, self.feat_names):
            nparts = feat.split(' ')
            # Identify the type of feature by the nparts
            if len(nparts) == 2:
                # PHENOLOGY feature identified
                if nparts[1] in self.phen:
                    # Phenology 1
                    fn = os.path.join(self.phendir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S1.hdf")

                    assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                    
                    param = feat[5:]

                    # No need to fill missing values, just read the values
                    feat_arr = read_from_hdf(fn, param, as_int16=True)  # Use HDF4 method
                    print(f"--{n:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)

                    # Fix values larget than 366
                    if param == 'SOS' or param == 'EOS' or param == 'LOS':
                        print(f' --Fixing {param}.')
                        feat_fixed = fix_annual_phenology(feat_arr)
                        feat_arr = feat_fixed[:]

                    # Fill missing data
                    if self.fill:
                        if param == 'SOS':
                            print(f'  --Filling {param}')
                            minimum = 0
                            sos = read_from_hdf(fn, 'SOS', as_int16=True)
                            eos = read_from_hdf(fn, 'EOS', as_int16=True)
                            los = read_from_hdf(fn, 'LOS', as_int16=True)
                            
                            sos_fixed = fix_annual_phenology(sos)
                            eos_fixed = fix_annual_phenology(eos)
                            los_fixed = fix_annual_phenology(los)
                            
                            # # Fix SOS values larger than 365
                            # sos_fixed = np.where(sos > 366, sos-365, sos)

                            # # Fix SOS values larger than 365, needs to be done two times
                            # eos_fixed = np.where(eos > 366, eos-365, eos)
                            # # print(np.min(eos_fixed), np.max(eos_fixed))
                            # if np.max(eos_fixed) > 366:
                            #     eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
                            #     print(f'  --Adjusting EOS again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')

                            filled_sos, filled_eos, filled_los =  fill_season(sos_fixed, eos_fixed, los_fixed, minimum,
                                                                            row_pixels=self.tile_rows,
                                                                            max_row=self.tile_rows,
                                                                            max_col=self.tile_cols,
                                                                            id=param,
                                                                            verbose=False)

                            feat_arr = filled_sos[:]
                        elif param == 'EOS':
                            print(f'  --Filling {param}')
                            feat_arr = filled_eos[:]
                        elif param == 'LOS':
                            print(f'  --Filling {param}')
                            feat_arr = filled_los[:]
                        elif param == 'DOP' or param == 'NOS':
                            # Day-of-peak and Number-of-seasons, use mode
                            print(f'  --Filling {param}')
                            feat_arr = fill_with_mode(feat_arr, 0, row_pixels=self.tile_rows, max_row=self.tile_rows, max_col=self.tile_cols, verbose=False)
                        elif param == 'GDR' or param == 'GUR' or param == 'MAX':
                            # GDR, GUR and MAX should be positive integers!
                            print(f'  --Filling {param}')
                            feat_arr = fill_with_int_mean(feat_arr, 0, var=param, verbose=False)
                        else:
                            # Other parameters? Not possible
                            print(f'  --Filling {param}')
                            ds = read_from_hdf(fn, param, as_int16=True)
                            feat_arr = fill_with_int_mean(ds, 0, var=param, verbose=False)

                    # Normalize or standardize
                    assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                    if self.normalize and not self.standardize:
                        feat_arr = normalize(feat_arr)
                    elif not self.normalize and self.standardize:
                        feat_arr = standardize(feat_arr)
                
                elif nparts[1] in self.phen2:
                    # Phenology 2
                    fn = os.path.join(self.phendir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S2.hdf")

                    # print(f"--{n:>3}: {feat} --> {fn}")
                    assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"

                    param = feat[5:]

                    # No need to fill missing values, just read the values
                    feat_arr = read_from_hdf(fn, param, as_int16=True)  # Use HDF4 method
                    print(f"--{n:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)

                    # Fix values larget than 366
                    if param == 'SOS' or param == 'EOS' or param == 'LOS':
                        print(f' --Fixing {param}.')
                        feat_fixed = fix_annual_phenology(feat_arr)
                        feat_arr = feat_fixed[:]

                    # Extract data and filter by training mask
                    if self.fill:
                        # IMPORTANT: Only a few pixels have a second season, thus dataset could
                        # have a huge amount of NaNs, filling will be restricted to replace a
                        # The missing values to NO_DATA
                        print(f'  --Filling {param}')
                        feat_arr = read_from_hdf(fn, param, as_int16=True)
                        feat_arr = np.where(feat_arr <= 0, self.nodata, feat_arr)
                    
                    # Normalize or standardize
                    assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                    if self.normalize and not self.standardize:
                        feat_arr = normalize(feat_arr)
                    elif not self.normalize and self.standardize:
                        feat_arr = standardize(feat_arr)
            else:
                # VI or SPECTRAL BAND feature
                if len(nparts) == 3:
                    month = nparts[0]
                    band = nparts[1]
                    fn = os.path.join(self.datadir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')
                elif len(nparts) == 4:
                    month = nparts[0]
                    band = nparts[2][1:-1]  # remove parenthesis
                    fn = os.path.join(self.datadir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')

                # print(f"--{n:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"

                # Extract data and filter
                feat_arr = read_from_hdf(fn, feat[4:], as_int16=True)  # Use HDF4 method
                print(f"--{n:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)

                ### Fill missing data
                if self.fill:
                    minimum = 0  # set minimum for spectral bands
                    # Look for VI's in the feature name, works for NDVI, EVI, or EVI2
                    band = get_band(feat)
                    if band in ['NDVI', 'EVI', 'EVI2']:
                        minimum = -10000  # minimum for VIs
                    feat_arr = fill_with_mean(feat_arr, minimum, var=band.upper(), verbose=False)

                # Normalize or standardize
                if self.normalize:
                    feat_arr = normalize(feat_arr)
            tile_features[:,:,n] = feat_arr
        return tile_features


    def features_by_season(self, fn_features, fn_features_season):
        """ Aggregates an HDF5 file with monthly features into another HDF5 file but with seasonal features 
        :param str fn_features: the input file name of the HDF5 file with monthly features
        :param str fn_features_season: the output file name of the HDF5 file with seasonal features
        """

        assert os.path.isfile(fn_features), f"ERROR: File not found: {fn_features}"

        h5_features = h5py.File(fn_features, 'r')
        h5_features_season = h5py.File(fn_features_season, 'w')

        # Group monthly data by season
        # seasons = {'SPR': ['MAR', 'APR', 'MAY'],
        #            'SUM': ['JUN', 'JUL', 'AUG'],
        #            'FAL': ['SEP', 'OCT', 'NOV'],
        #            'WIN': ['JAN', 'FEB', 'DEC']}
        seasons = {'SPR': ['APR', 'MAY', 'JUN'],
                'SUM': ['JUL', 'AUG', 'SEP'],
                'FAL': ['OCT', 'NOV', 'DEC'],
                'WIN': ['JAN', 'FEB', 'MAR']}
        # Get the unique bands and variables from feature names
        bands = []
        vars = []
        for feature in self.feat_names:
            band = get_band(feature)
            var = feature.split(" ")[-1]
            if band not in bands:
                bands.append(band)
            if var not in vars:
                vars.append(var)
        
        # Group feature names by season -> band -> variable -> month
        season_feats = {}
        for season in list(seasons.keys()):
            print(f"  AGGREGATING: {season}")
            for band in bands:
                band = band.upper()
                for var in vars:
                    for feat_name in self.feat_names:
                        # Split the feature name to get band and month
                        ft_name_split = feat_name.split(' ')
                        if len(ft_name_split) == 2:
                            # Ignore phenology, do not group
                            continue
                        elif len(ft_name_split) == 3:
                            feat_band = ft_name_split[1]
                        elif len(ft_name_split) == 4:
                            feat_band = ft_name_split[2][1:-1]  # remove parenthesis
                            feat_band = feat_band.upper()
                        feat_var = ft_name_split[-1]
                        feat_month = ft_name_split[0]

                        for month in seasons[season]:
                            if band == feat_band and var == feat_var and month == feat_month:
                                season_key = season + ' ' + band + ' ' + var
                                if season_feats.get(season_key) is None:
                                    season_feats[season_key] = [feat_name]
                                else:
                                    season_feats[season_key].append(feat_name)
                                # print(f"  -- {season} {band:>5} {var:>5}: {feat_name}")
        self.feat_indices_season = []
        self.feat_names_season = []
        feat_num = 0

        # Calculate averages of features grouped by season
        for key in list(season_feats.keys()):
            print(f"  *{key:>15}:")
            for i, feat_name in enumerate(season_feats[key]):
                print(f"   -Adding {feat_num}: {feat_name}")

                # Add the data
                if i == 0:
                    # Initialize array to hold average
                    feat_arr = h5_features[feat_name][:]
                    # print(f"   -{feat_arr.dtype}")
                else:
                    # Add remaining months
                    feat_arr += h5_features[feat_name][:]
                    # print(f"   -{feat_arr.dtype}")
                
            # Average, force to int16 type
            # feat_arr /= len(season_feats[key])
            feat_arr = np.round(np.round(feat_arr).astype(np.int16) / np.int16(len(season_feats[key]))).astype(np.int16)

            h5_features_season.create_dataset(key, feat_arr.shape, data=feat_arr)

            self.feat_indices_season.append(feat_num)
            self.feat_names_season.append(key)

            feat_num += 1

        # Add PHEN features directly, no aggregation by season
        for feat_name in self.feat_names:
            if feat_name[:4] == 'PHEN':
                print(f"   -Adding {feat_num}: {feat_name}")

                # Extract data
                feat_arr = h5_features[feat_name][:]
                # print(f"   -{feat_arr.dtype}")

                # Write data
                h5_features_season.create_dataset(feat_name, feat_arr.shape, data=feat_arr)

                self.feat_indices_season.append(feat_num)
                self.feat_names_season.append(feat_name)

                feat_num += 1
        
        print(f"File: {fn_features_season} created successfully.")


    def read_tiles_extent(self):
        """ Put the extent of tiles into a dictionary 
        Extent format is a dictionary with N, S, W, and E boundaries
        """
        self.tiles_extent = {}
        with open(self.fn_tiles, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # print(row)
                row_dict = {}
                for item in row[1:]:
                    itemlst = item.split('=')
                    row_dict[itemlst[0].strip()] = int(float(itemlst[1]))
                self.tiles_extent[row[0]] = row_dict
        # print(tiles_extent)


    def create_tile_dataset(self, list_tiles: str, **kwargs) -> None:
        save_labels_raster = kwargs.get('save_labels_raster', False)
        save_features = kwargs.get('save_features', False)
        by_season = kwargs.get('by_season', False)

        if by_season:
            # This option in mandatory in this case
            save_features = True
            
        self.set_tiles(list_tiles)

        # First, get tiles extent
        self.read_tiles_extent()

        assert self.tiles is not None, "List of tiles is empty (None)."

        # In case sampling was executed in a previous run
        if self.training_mask is None:
            self.read_training_mask()
        if self.training_labels is None:
            self.read_training_labels()

        self.no_data_arr = np.where(self.landcover > 0, 1, self.nodata)  # 1=data, 0=NoData
        # self.no_data_arr = np.where(self.landcover > 0, 1, self.NO_DATA)  # 1=data, 0=NoData
        # Keep train mask values only in pixels with data, remove NoData
        print(f" Train mask shape: {self.training_mask.shape}")
        self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.nodata)
        # self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.NO_DATA)

        # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
        print(f"  --no_data_arr={self.no_data_arr.dtype}, training_mask={self.training_mask.dtype} ")
        print(f'  --Training pixels: {(self.training_mask == 1).sum()}')
        print(f'  --Testing pixels: {(self.training_mask == 0).sum()}')

        for tile in self.tiles:
            print(f"\nProcessing tile: {tile}")
            
            # Create new directories
            labels_path = os.path.join(self.cwd, 'data/inegi', tile)
            feat_path = os.path.join(self.cwd, 'features', self.phenobased, tile)
            if not os.path.exists(labels_path) and save_labels_raster:
                print(f"\nCreating new path: {labels_path}")
                os.makedirs(labels_path)
            if not os.path.exists(feat_path) and save_features:
                print(f"\nCreating new path: {feat_path}")
                os.makedirs(feat_path)

            # Create new file names
            fn_base = os.path.basename(self.fn_landcover)
            fn_tile = os.path.join(labels_path, f"{fn_base[:-4]}_{tile}.tif")
            
            # Extent will be N-S, and W-E boundaries
            merged_ext = {}
            merged_ext['W'], xres, _, merged_ext['N'], _, yres = [int(x) for x in self.geotransform]
            merged_ext['E'] = merged_ext['W'] + self.ncols*xres
            merged_ext['S'] = merged_ext['N'] + self.nrows*yres
            print(merged_ext)

            # Calculate slice coodinates to extract the tile
            tile_ext = self.tiles_extent[tile]

            # Get row for Nort and South and column for West and East
            nrow = (tile_ext['N'] - merged_ext['N'])//yres
            srow = (tile_ext['S'] - merged_ext['N'])//yres
            wcol = (tile_ext['W'] - merged_ext['W'])//xres
            ecol = (tile_ext['E'] - merged_ext['W'])//xres
            print(f"{tile}: N={nrow} S={srow} W={wcol} E={ecol}")
            
            tile_geotransform = (tile_ext['W'], xres, 0, tile_ext['N'], 0, yres)
            print(f"Tile geotransform: {tile_geotransform}")

            # Slice the labels from raster
            tile_landcover = self.landcover[nrow:srow, wcol:ecol]
            print(f"Slice: {nrow}:{srow}, {wcol}:{ecol} {tile_landcover.shape}")
            if save_labels_raster:
                # Save the sliced data into a new raster
                print(f"Writing: {fn_tile}")
                create_raster_proj(fn_tile, tile_landcover, self.spatial_reference, tile_geotransform)
            
            # Slice the training mask and the NoData mask
            tile_training_mask = self.training_mask[nrow:srow, wcol:ecol]
            tile_nodata = self.no_data_arr[nrow:srow, wcol:ecol]

            # Generate features
            tile_features = self.generate_tile_features(tile, tile_nodata, fill=False)

            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_labels = os.path.join(feat_path, f"labels_{tile}.h5")

            # Save the features
            if save_features:
                print(f"Saving {tile} features...")
                # Create (large) HDF5 files to hold all features
                h5_features = h5py.File(fn_tile_features, 'w')
                h5_labels = h5py.File(fn_tile_labels, 'w')

                # Save the training and testing labels
                h5_labels.create_dataset('all', (self.tile_rows, self.tile_cols), data=tile_landcover, dtype=self.landcover.dtype)
                h5_labels.create_dataset('training_mask', (self.tile_rows, self.tile_cols), data=tile_training_mask, dtype=self.landcover.dtype)
                h5_labels.create_dataset('no_data_mask', (self.tile_rows, self.tile_cols), data=tile_nodata, dtype=self.landcover.dtype)

                for n, feature in zip(self.feat_indices, self.feat_names):
                    h5_features.create_dataset(feature, (self.tile_rows, self.tile_cols), data=tile_features[:,:,n])

                    # INTEGER CONVERSION WAS FIXED AT READING TIME FROM HDF4 (SEE read_from_hdf FUNCTION)
                    # single_tile_features = np.round(tile_features[:,:,n]).astype(np.int16)
                    # h5_features.create_dataset(feature, (self.tile_rows, self.tile_cols), data=single_tile_features, dtype=np.int16)
                    
                    # Define the band data type to optimize storage
                    # feat_dtype = int
                    # band = get_band(feature)
                    # var = feature.split(" ")[-1]
                    # # Save features
                    # if band in ['GUR', 'GDR', 'GUR2', 'GDR2'] or var == 'STDEV':
                    #     print(f"  --Saving {feature} as floating point.")
                    #     feat_dtype = float
                        
                    # h5_features.create_dataset(feature, (self.tile_rows, self.tile_cols), data=tile_features[:,:,n], dtype=feat_dtype)
            
            if by_season:
                # Aggregate features by season
                # WARNING! Requires HDF5 files to be created first!
                fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")
                self.features_by_season(fn_tile_features, fn_tile_feat_season)
            
        # Finish creating feature dataset per tile
    

class LandCoverClassifier():

    """ Trains a single random forest per tile """

    fmt = '%Y_%m_%d-%H_%M_%S'

    def __init__(self, cwd, list_tiles, fn_landcover, phenobased):
        self.cwd = cwd
        self.list_tiles = list_tiles
        self.fn_landcover = fn_landcover
        self.phenobased = phenobased

        self.features = None
        self.nrows = None
        self.ncols = None

        self.train_mask = None
        self.nan_mask = None
        self.X = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.landcover = None
        self.epsg = None
        self.nodata = None
        self.metadata = None
        self.projection = None
        self.geotransform = None

        self.spatial_reference = None

        self.read_raster_parameters()
        # self.read_spatial_reference()


    def read_raster_parameters(self):
        """ Reads raster parameters """

        assert os.path.isfile(self.fn_landcover) is True, f"ERROR: File not found! {self.fn_landcover}"
        
        self.landcover, self.nodata, self.geotransform, self.spatial_reference = open_raster(self.fn_landcover)
        # self.landcover, self.nodata, self.metadata, self.geotransform, self.projection, self.epsg = open_raster(self.fn_landcover)
        self.landcover = self.landcover.astype(np.int8)  # convert to byte (int8)
        self.landcover = self.landcover.filled(0) # replace masked constant "--" with zeros
        # self.epsg = np.uint16(self.epsg)  # uint16 == ushort
        self.nodata = np.int8(self.nodata)

        self.nrows = self.landcover.shape[0]
        self.ncols = self.landcover.shape[1]
        
        print(f'  -- Opening raster : {self.fn_landcover}')
        print(f'  ---- Spatial ref. : {self.spatial_reference}')
        print(f'  ---- NoData       : {self.nodata}')
        print(f'  ---- Columns      : {self.ncols}')
        print(f'  ---- Rows         : {self.nrows}')
        print(f'  ---- Geotransform : {self.geotransform}')
        # print(f'  ---- Projection   : {self.projection}')
        # print(f'  ---- EPSG         : {self.epsg}')
        print(f'  ---- Type         : {self.landcover.dtype}')

    def read_spatial_reference(self):
        with open(os.path.join(self.cwd, 'parameters', 'spatial_reference'), 'r') as f:
            self.spatial_reference = f.read()
        print(f"Setting spatial reference: {self.spatial_reference}")


    def classify_by_tile(self, tiles=None):

        if tiles is not None:
            print("WARNING!: Overriding list of tiles!")
            self.list_tiles = tiles

        start = datetime.now()

        # Create directory to save results
        results_path = os.path.join(self.cwd, 'results', self.phenobased, f"{datetime.strftime(start, self.fmt)}")
        if not os.path.exists(results_path):
            print(f"\nCreating new path: {results_path}")
            os.makedirs(results_path)

        for tile in self.list_tiles:
            print(f"\n *** Running classifier in tile {tile} ***\n")

            # Configure files names
            tile_results_path = os.path.join(results_path, tile)
            if not os.path.exists(tile_results_path):
                print(f"\nCreating new path: {tile_results_path}")
                os.makedirs(tile_results_path)

            fn_save_model = os.path.join(tile_results_path, "rf_model.pkl")
            fn_save_importance = os.path.join(tile_results_path, "rf_feat_importance.csv")
            fn_save_crosstab_train = os.path.join(tile_results_path, "rf_crosstab_train.csv")
            fn_save_crosstab_test = os.path.join(tile_results_path, "rf_crosstab_test.csv")
            fn_save_crosstab_test_mask = fn_save_crosstab_test[:-4] + '_mask.csv'
            fn_save_conf_tbl = os.path.join(tile_results_path, "rf_confussion_table.csv")
            fn_save_report = os.path.join(tile_results_path, "rf_classif_report.txt")
            fn_save_preds_fig = os.path.join(tile_results_path, "rf_predictions.png")
            fn_save_preds_raster = os.path.join(tile_results_path, "rf_predictions.tif")
            fn_save_preds_h5 = os.path.join(tile_results_path, "rf_predictions.h5")
            fn_save_params = os.path.join(tile_results_path, "rf_parameters.csv")
            fn_save_conf_fig = fn_save_conf_tbl[:-4] + '.png'
            
            fn_colormap = self.cwd + 'parameters/qgis_cmap_landcover_CBR_viri_grp11.clr'
            
            # Look for feature files in each tile
            feat_path = os.path.join(self.cwd, 'features', self.phenobased, tile)
            fn_tile_labels = os.path.join(feat_path, f"labels_{tile}.h5")
            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")

            assert os.path.isfile(fn_tile_feat_season) or os.path.isfile(fn_tile_features), "ERROR: features files not found!"
            assert os.path.isfile(fn_tile_labels), "ERROR: labels file not found!" 

            if os.path.isfile(fn_tile_feat_season):
                # Look for seasonal features first
                print(f"Found seasonal features: {fn_tile_feat_season}.")
                fn_tile_features = fn_tile_feat_season  # point to features by season
            elif os.path.isfile(fn_tile_features):
                print(f"Found monthly features: {fn_tile_features}")

            # Read the features to initialize
            with h5py.File(fn_tile_features, 'r') as h5_features:
                self.features = [key for key in h5_features.keys()]
                for i, feature in enumerate(self.features):
                    print(f" Feature {i}: {feature}")
                # Set rows and columns
                dummy = h5_features[self.features[0]][:]
                self.nrows, self.ncols = dummy.shape
                del dummy
            
            # # TODO: Change to uint8? Requires generating all NDVI and EVI datasets!!!
            # x_train = np.empty((self.nrows, self.ncols, len(self.features)), dtype=np.int16)
            # y = np.empty((self.nrows, self.ncols), dtype=np.int8)
            # train_mask = np.empty((self.nrows, self.ncols), dtype=np.int8)
            # nan_mask = np.empty((self.nrows, self.ncols), dtype=np.int8)

            # Read the labels
            print("Reading labels...")
            with h5py.File(fn_tile_labels, 'r') as h5_labels:
                self.y = h5_labels['all'][:]
                self.train_mask = h5_labels['training_mask'][:]
                self.nan_mask = h5_labels['no_data_mask'][:]

            print(f"y: {self.y.dtype}")
            print(f"train_mask: {self.train_mask.dtype}")
            print(f"nan_mask: {self.nan_mask.dtype}")
            self.train_mask = self.train_mask.flatten()
            self.nan_mask = self.nan_mask.flatten()
            self.y = self.y.flatten()  # flatten by appending rows, each value will correspod to a row in X

            # Read the features, for real
            x = np.empty((self.nrows, self.ncols, len(self.features)), dtype=np.int16)
            with h5py.File(fn_tile_features, 'r') as h5_features:
                # Get the data from the HDF5 files
                for i, feature in enumerate(self.features):
                    x[:,:,i] = h5_features[feature][:]

            # Reshape x_train into a 2D-array of dimensions: (rows*cols, bands)
            print("Reading features...")
            x_temp = x.copy()
            self.X = np.empty((self.nrows*self.ncols, len(self.features)), dtype=np.int16)
            i = 0
            for row in range(self.nrows):
                for col in range(self.nrows):
                    # print(f'row={row}, col={col}: {X_temp[:,row,col]} {X_temp[:,row,col].shape}')
                    # if row%500 == 0 and col%100 == 0:
                    #     print(f'{i} row={row}, col={col}: {x_temp[row, col,:]} {x_temp[row, col,:].shape}')
                    # Place all bands from a pixel into a row
                    self.X[i,:] = x_temp[row, col,:]
                    i += 1
            # Delete temporal variables
            del x_temp
            del x

            print("Creating training and testing datasets...")
            self.x_train = self.X[self.train_mask > 0]
            self.y_train = self.y[self.train_mask > 0]

            # Create a TESTING MASK: Select on the valid region only (discard NoData pixels)
            self.test_mask = np.logical_and(self.train_mask == 0, self.nan_mask == 1)
            self.x_test = self.X[self.test_mask]
            self.y_test = self.y[self.test_mask]

            print(f'  --x_train shape={self.x_train.shape}')
            print(f'  --y_train shape={self.y_train.shape}')
            print(f'  --x_test shape={self.x_test.shape}')
            print(f'  --y_test shape={self.y_test.shape}')
            print(f'  --X shape={self.X.shape}')
            print(f'  --y shape={self.y.shape}')

            tr_lbl, tr_fq = np.unique(self.train_mask, return_counts=True)
            df_mask = pd.DataFrame({'TrainVal': tr_lbl, 'TrainFq': tr_fq})
            df_mask.loc['Total'] = df_mask.sum(numeric_only=True, axis=0)
            print(df_mask)

            # Check labels between train and test are the same
            tr_lbl, tr_fq = np.unique(self.y_train, return_counts=True)
            df = pd.DataFrame({'TrainLbl': tr_lbl, 'TrainFreq': tr_fq})
            df.loc['Total'] = df.sum(numeric_only=True, axis=0)
            print(df)

            ### TRAIN THE RANDOM FOREST
            print(f'  {datetime.strftime(datetime.now(), self.fmt)}: starting Random Forest training')
            print('  Creating the model')
            start_train = datetime.now()

            rf_trees = 250
            rf_depth = None
            rf_jobs = 64
            # class_weight = {class_label: weight}
            rf_weight = None

            rf = RandomForestClassifier(n_estimators=rf_trees,
                                        oob_score=True,
                                        max_depth=rf_depth,
                                        n_jobs=rf_jobs,
                                        class_weight=rf_weight,
                                        verbose=1)

            print(f'  {datetime.strftime(datetime.now(), self.fmt)}: fitting the model...')
            rf = rf.fit(self.x_train, self.y_train)

            # Save trained model
            print("Saving trained model...")
            with open(fn_save_model, 'wb') as f:
                pickle.dump(rf, f)

            print(f'  --OOB prediction of accuracy: {rf.oob_score_ * 100:0.2f}%')

            # feat_n = []
            feat_list = []
            feat_imp = []
            for feat, imp in zip(self.features, rf.feature_importances_):
                # print(f'  --{feat_index[b]:>15}: {imp:>0.6f}')
                # feat_n.append(b)
                feat_list.append(feat)
                feat_imp.append(imp)
            
            feat_importance = pd.DataFrame({'Feature': feat_list, 'Importance': feat_imp})
            feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
            print("Feature importance: ")
            print(feat_importance.to_string())
            feat_importance.to_csv(fn_save_importance)

            end_train = datetime.now()
            training_time = end_train - start_train
            print(f'  {datetime.strftime(end_train, self.fmt)}: training finished in {training_time}.')

            # Predict on the rest of the image, using the fitted Random Forest classifier
            start_pred = datetime.now()
            print(f'  {datetime.strftime(start_pred, self.fmt)}: making predictions')

            y_pred_train = rf.predict(self.x_train)

            # A crosstabulation to see class confusion for TRAINING
            df_tr = pd.DataFrame({'truth': self.y_train, 'predict': y_pred_train})
            crosstab_tr = pd.crosstab(df_tr['truth'], df_tr['predict'], margins=True)
            crosstab_tr.to_csv(fn_save_crosstab_train)

            y_pred_test = rf.predict(self.x_test)

            # A crosstabulation to see class confusion for TESTING (MASKED)
            df_ts = pd.DataFrame({'truth': self.y_test, 'predict': y_pred_test})
            crosstab_ts = pd.crosstab(df_ts['truth'], df_ts['predict'], margins=True)
            crosstab_ts.to_csv(fn_save_crosstab_test_mask)

            y_pred = rf.predict(self.X)

            # A crosstabulation to see class confusion for TESTING (COMPLETE MAP)
            df = pd.DataFrame({'truth': self.y, 'predict': y_pred})
            crosstab = pd.crosstab(df['truth'], df['predict'], margins=True)
            crosstab.to_csv(fn_save_crosstab_test)

            print(f'  --y_pred_train shape:', y_pred_train.shape)
            print(f'  --y_pred_test shape:', y_pred_test.shape)
            print(f'  --y_pred shape:', y_pred.shape)

            accuracy = accuracy_score(self.y_test, y_pred_test)
            print(f'  --Accuracy score (testing dataset): {accuracy}')

            cm = confusion_matrix(self.y_test, y_pred_test)
            with open(fn_save_conf_tbl, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for single_row in cm:
                    writer.writerow(single_row)
                    # print(single_row)
            
            report = classification_report(self.y_test, y_pred_test, )
            print('  Classification report')
            print(report)
            with open(fn_save_report, 'w') as f:
                f.write(report)
            
            end_pred = datetime.now()
            pred_time = end_pred - start_pred
            print(f'  {datetime.strftime(datetime.now(), self.fmt)}: prediction finished in {pred_time}')

            # Reshape the classification map into a 2D array again to show as a map
            y_pred = y_pred.reshape((self.nrows, self.ncols))

            print(f'  --y_pred (re)shape:', y_pred.shape)

            print(f"  --Pred train: {np.unique(y_pred_train)}")
            print(f"  --Pred test: {np.unique(y_pred_test)}")
            print(f"  --Pred (all): {np.unique(y_pred)}")

            # Plot the land cover map of the predictions for y and the whole area
            plot_array_clr(y_pred, fn_colormap, savefig=fn_save_preds_fig, zero=True)  # zero=True, zeros removed with mask?

            # Save predicted land cover classes into a GeoTIFF
            # create_raster_proj(fn_save_preds_raster, y_pred, self.spatial_reference, self.geotransform)
            tile_geotransform = []
            create_raster(fn_save_preds_raster, y_pred, self.spatial_reference, tile_geotransform)

            # Save predicted land cover classes into a HDF5 file
            with h5py.File(fn_save_preds_h5, 'w') as h5_preds:
                h5_preds.create_dataset("predictions", (self.nrows, self.ncols), data=y_pred)

            with open(fn_save_params, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['Start', start])
                writer.writerow(['CWD', cwd])
                writer.writerow(['Format', self.fmt])
                writer.writerow(['x_train shape', f'{self.x_train.shape}'])
                writer.writerow(['y_train shape', f'{self.y_train.shape}'])
                writer.writerow(['x_test shape', f'{self.X.shape}'])
                writer.writerow(['y_test shape', f'{self.y.shape}'])
                writer.writerow(['MODEL:', 'RandomForestClassifier'])
                writer.writerow([' Estimators', rf_trees])
                writer.writerow([' Max depth', rf_depth])
                writer.writerow([' Jobs', rf_jobs])
                writer.writerow([' Class weight:', rf_weight])
                writer.writerow([' OOB prediction of accuracy', f'{rf.oob_score_}' ])
                writer.writerow([' Accuracy score', f'{accuracy}' ])
                writer.writerow([' Start training', f'{start_train}'])
                writer.writerow([' End training', f'{end_train}'])
                writer.writerow([' Training time', f'{training_time}'])
                writer.writerow([' Start testing (prediction)', start_pred])
                writer.writerow([' End testing (prediction)', end_pred])
                writer.writerow([' Testing time (prediction)', pred_time])

            # Plot the confusion table
            n_classes = len(np.unique(y_pred_test))
            self.land_cover_conf_table(fn_save_conf_tbl, n_classes, savefig=fn_save_conf_fig, normalize=False)

            self.free_memory()
            del df_tr
            del df_ts
            del df
            del y_pred_train
            del y_pred_test
            del y_pred

            gc.collect

            print(f'  {datetime.strftime(datetime.now(), self.fmt)}: finished in {datetime.now() - start}')
            print(f'  Done with tile {tile}')


    def free_memory(self):
        # Free memory
        del self.X
        del self.y
        del self.x_train
        del self.y_train
        del self.x_test
        del self.y_test
        del self.train_mask
        del self.nan_mask


    def land_cover_conf_table(self, fn_table, n_classes, **kwargs):
        """ Plots a land cover confussion table """
        _normalize = kwargs.get('normalize', False)
        _title = kwargs.get('title', '')
        _savefig = kwargs.get('savefig', '')
        _dpi = kwargs.get('dpi', 300)

        values = np.array(pd.read_csv(fn_table, header=None))
        if _normalize:
            values = (values - np.min(values)) / (np.max(values) - np.min(values))

        land_cover = [x for x in range(0, n_classes)]

        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.imshow(values)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(land_cover)), labels=land_cover)
        ax.set_yticks(np.arange(len(land_cover)), labels=land_cover)
        ax.grid(False)

        # Loop over data dimensions and create text annotations.
        for i in range(len(land_cover)):
            for j in range(len(land_cover)):
                if _normalize:
                    text = ax.text(j, i, f'{values[i, j]:0.2f}', ha="center", va="center", color="w", fontsize='x-small')
                else:
                    text = ax.text(j, i, f'{values[i, j]:0.0f}', ha="center", va="center", color="w", fontsize='x-small')
                
        if _title != '':
            ax.set_title(_title)
        if _savefig != '':
            plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
        else:
            plt.show()
        plt.close()


def get_band(feature_name: str) -> str:
    band = ''
    nparts = feature_name.split(' ')
    if len(nparts) == 2:
        band = nparts[1]
    elif len(nparts) == 3:
        band = nparts[1]
    elif len(nparts) == 4:
        band = nparts[2][1:-1]  # remove parenthesis
    return band


def open_raster_old(filename: str) -> tuple:
    """ Open a GeoTIFF raster and return a numpy array

    :param str filename: the file name of the GeoTIFF raster to open
    :return raster: a masked array with NoData values masked out
    """

    dataset = gdal.OpenEx(filename)

    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    raster_array = dataset.ReadAsArray()
    projection = dataset.GetProjection()
    proj = osr.SpatialReference(wkt=dataset.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    # print(epsg)
    
    # Mask 'NoData' values
    raster = np.ma.masked_values(raster_array, nodata)

    # Clean
    del(dataset)
    del(raster_array)
    gc.collect()

    return raster, nodata, metadata, geotransform, projection, epsg


def open_raster(filename: str) -> tuple:
    dataset = gdal.OpenEx(filename)

    # metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    raster_array = dataset.ReadAsArray()
    spatial_ref = osr.SpatialReference(wkt=dataset.GetProjection())
    
    # Mask 'NoData' values
    raster = np.ma.masked_values(raster_array, nodata)

    # Clean
    del(dataset)
    del(raster_array)
    gc.collect()

    return raster, nodata, geotransform, spatial_ref


def show_raster(filename: str, **kwargs) -> None:
    
    _savefigs = kwargs.get('savefigs', '')
    _cmap = kwargs.get('savefigs', 'viridis')
    _dpi = kwargs.get('dpi', 300)
    _size = kwargs.get('figsize', (12,12))
    _verbose = kwargs.get('verbose', False)

    print(f'\n  Openning raster: {filename}')
    
    # Open the raster, read it as array and get the geotransform
    raster_arr, nd, meta, gt = open_raster(filename)

    # Get the raster extent
    rows, cols = raster_arr.shape
    ulx, xres, _, uly, _, yres = gt
    extent = [ulx, ulx + xres*cols, uly, uly + yres*rows]

    if _verbose:
        print(f'  --Metadata: {meta}')
        print(f'  --NoData  : {nd}')
        print(f'  --Columns : {cols}')
        print(f'  --Rows    : {rows}')
        print(f'  --Extent  : {extent}')
    
    # Display with matplotlib
    if _savefigs != '':
        plt.figure(figsize=_size)
        plt.imshow(raster_arr, cmap=_cmap)
        plt.colorbar()
        # plt.show()
        plt.savefig(_savefigs, bbox_inches='tight', dpi=_dpi)
        plt.close()


def create_raster_old(filename: str, data: np.ndarray, epsg: int, geotransform: list, **kwargs) -> None:
    """ Create a raster (GeoTIFF) from a numpy array, by default uses byte (int8) format """
    _as_byte = kwargs.get('as_byte', True)
    _verbose = kwargs.get('verbose', False)

    if type(epsg) is not int:
        epsg = int(epsg)

    driver_gtiff = gdal.GetDriverByName('GTiff')

    rows, cols = data.shape

    ds_create = driver_gtiff.Create(filename, xsize=cols, ysize=rows, bands=1)
    if _as_byte:
        ds_create = driver_gtiff.Create(filename, xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Byte)

    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds_create.SetProjection(srs.ExportToWkt())

    # Set the geotransform
    ds_create.SetGeoTransform(geotransform)

    if _verbose:
        print(f'  --Type of driver: {type(driver_gtiff)}')
        print(f'  --As byte (int8): {_as_byte}')
        print(f'  --{ds_create.GetProjection()}')
        print(f'  --{ds_create.GetGeoTransform()}')

    ds_create.GetRasterBand(1).WriteArray(data)  # write the array to the raster
    ds_create.GetRasterBand(1).SetNoDataValue(0)  # set the no data value
    ds_create = None  # properly close the raster


def create_raster(filename: str, data: np.ndarray, spatial_ref: str, geotransform: list, **kwargs) -> None:
    """ Creates a raster (GeoTIFF) from a numpy array using a custom spatial reference """
    _verbose = kwargs.get('verbose', False)

    driver_gtiff = gdal.GetDriverByName('GTiff')

    rows, cols = data.shape

    ds_create = driver_gtiff.Create(filename, xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Byte)

    # Set the custom spatial reference
    srs = osr.SpatialReference()
    if type(spatial_ref) is int:
        srs.ImportFromEPSG(spatial_ref)
    elif type(spatial_ref) is str:
        # srs.ImportFromProj4(spatial_ref)
        srs.ImportFromWkt(spatial_ref)
    elif type(spatial_ref) is osr.SpatialReference:
        srs = spatial_ref
    # print(f"  Using spatial reference: {spatial_ref} to create raster.")
    # print(srs.ExportToWkt())
    ds_create.SetProjection(srs.ExportToWkt())

    # Set the geotransform
    ds_create.SetGeoTransform(geotransform)

    if _verbose:
        print(f'  --Creating dataset: {filename}')
        print(f'  -- Type of driver: {type(driver_gtiff)}')
        print(f'  -- Spatial reference: {spatial_ref}')
        print(f'  -- Spatial ref. WKT: {srs.ExportToWkt()}')
        print(f'  -- {ds_create.GetProjection()}')
        print(f'  -- {ds_create.GetGeoTransform()}')
    
    ds_create.GetRasterBand(1).WriteArray(data)  # write the array to the raster
    ds_create.GetRasterBand(1).SetNoDataValue(0)  # set the no data value
    ds_create = None  # properly close the raster


def fix_annual_phenology(data: np.ndarray) -> np.ndarray:
    MAX_ITERS = 10
    iter = 0
    while np.max(data) > 366 or iter >= MAX_ITERS:
        data = np.where(data > 366, data-365, data)
        iter += 1
    return data


def filter_raster(raster: np.ndarray, filters: list, **kwargs) -> np.ndarray:
    """ Apply pixel-wise filters to the raster, find the desired value in each
    pixel of the raster and return

    :param masked array raster: the raster to filter, integer valued
    :param list filter: integer values (from binary filters) fo search
    :return masked array: pixels selected by the filter """

    _verbose = kwargs.get('verbose', False)
    _savefigs = kwargs.get('savefigs', '')

    # Get unique values in raster
    unique_values = np.unique(np.ma.getdata(raster))
    print(f'  --Unique values in raster: {unique_values}')
    
    # Create an array to save selected pixels
    pixel_qa_mask = np.zeros(raster.shape, dtype=np.int16)
    # Apply same mask as 'raster' for 'NoData'
    pixel_qa_mask = np.ma.masked_array(pixel_qa_mask, mask=np.ma.getmask(raster))

    # Apply each filter
    for i, filter in enumerate(filters):
        
        # Apply only filters in the raster's unique values
        if filter not in unique_values:
            if _verbose:
                print(f'  --Filter {filter} ({i+1} of {len(filters)}) not found. Skipping...')
            continue

        # filtered_qa = raster & filter # Will include similar binary values, avoid
        filtered_qa = np.equal(raster, filter)  # Get the exact value

        if _verbose:
            print(f'  --Filter: {filter} ({i+1} of {len(filters)})')
            # Get unique values
            filtered_arr = np.ma.getdata(filtered_qa)
            uniques = np.unique(filtered_arr)
            # uniques = np.unique(filtered_qa)
            print(f'  --{len(uniques)} unique value(s): {uniques}')

        # Create the mask of selected pixels, accumulate all filters
        pixel_qa_mask += filtered_qa.astype(dtype=bool)

        # Save the individual images for each filter
        if _savefigs != '':
            plt.figure(figsize=(12,12))
            plt.imshow(filtered_qa, interpolation='none', resample=False)
            plt.colorbar()
            plt.savefig(f'{_savefigs}_filter{i}.png', bbox_inches='tight', dpi=600)
            plt.close()
    
    # Image with all filters (that were found) together
    if _savefigs != '':
        plt.figure(figsize=(12,12))
        plt.imshow(pixel_qa_mask.astype(dtype=bool), interpolation='none', resample=False)
        plt.colorbar()
        plt.savefig(f'{_savefigs}_filter_all.png', bbox_inches='tight', dpi=600)
        plt.close()

    return pixel_qa_mask


# Bitmask functions: get_bit, get_normalized_bit
def get_bit(value: int, bit_index: int) -> int:
    """ Retrieves the bit value of a given position by index """
    return value & (1 << bit_index)


def get_normalized_bit(value: int, bit_index: int) -> int:
    """ Retrieves the normalized bit value (0-1) of a given position by index """
    return (value >> bit_index) & 1


def save_plot_array(array: np.ndarray, filename: str, **kwargs) -> None:
    _path = kwargs.get('path', 'results/')
    """ Saves a plot of an array with a colormap """
    _suffix = kwargs.get('suffix', '')
    _title = kwargs.get('title', '')
    plt.figure(figsize=(12,12))
    plt.imshow(array, interpolation='none', resample=False)
    plt.title(filename + ' ' + _title)
    plt.colorbar()
    plt.savefig('results/' + filename + _suffix + '.png', bbox_inches='tight', dpi=600)
    plt.close()


def save_cmap_array(array: np.ndarray, filename: str, colors_dict: dict, labels: list, **kwargs) -> None:
    """ Saves a plot of an array with unique values and labels in colormap """
    _path = kwargs.get('path', 'results/')
    _suffix = kwargs.get('suffix', '')
    _title = kwargs.get('title', '')

    assert len(colors_dict) == len(labels), "Mismatch in colors and labels lists"

    # Create color map from list of colors
    cmap = ListedColormap([colors_dict[x] for x in colors_dict.keys()])
    len_lab = len(labels)

    # Prepare bins for normalizer
    norm_bins = np.sort([*colors_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # print(norm_bins)

    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot figure
    fig,ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(array, interpolation='none', resample=False, cmap=cmap, norm=norm)
    plt.title(filename + ' ' + _title)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz)
    fig.savefig(_path + filename + _suffix + '.png', bbox_inches='tight', dpi=600)
    plt.close()


def array_stats(title: str, array: np.ndarray) -> None:
    print(f'  --{title}  max: {array.max():.2f}, min:{array.min():.2f} avg: {array.mean():.4f} std: {array.std():.4f}')


# def read_from_hdf(filename: str, var: str, as_int16: bool=False) -> np.ndarray:
#     """ Reads a HDF4 file and return its content as numpy array
#     :param str filename: the name of the HDF4 file
#     :param str var: the dataset (or variable) to select
#     :return: a numpy array
#     """
#     data_raster = filename
#     hdf_bands = SD(data_raster, SDC.READ)  # HDF4 file with the land cover data sets
#     # print(f'Datasets: {hdf_bands.datasets()}')
#     values_arr = hdf_bands.select(var)  # Open dataset

#     # Dump the info into a numpy array
#     data_arr = np.array(values_arr[:])

#     # Close dataset
#     values_arr.endaccess()
#     # Close file
#     hdf_bands.end()

#     if as_int16:
#         data_arr = np.round(data_arr).astype(np.int16)

#     return data_arr

def read_from_hdf(filename: str, var: str, dtype: np.dtype = None) -> np.ndarray:
        """ Reads a HDF4 file and return its content as numpy array
        :param str filename: the name of the HDF4 file
        :param str var: the dataset (or variable) to select
        :return: a numpy array
        """
        data_raster = filename
        hdf_bands = SD(data_raster, SDC.READ)  # HDF4 file with the land cover data sets
        # print(f'Datasets: {hdf_bands.datasets()}')
        values_arr = hdf_bands.select(var)  # Open dataset

        # Dump the info into a numpy array
        data_arr = np.array(values_arr[:])

        # Close dataset
        values_arr.endaccess()
        # Close file
        hdf_bands.end()

        if dtype is not None:
            data_arr = np.round(data_arr).astype(dtype)

        return data_arr


def create_qa(pixel_qa: np.ndarray, sr_aerosol: np.ndarray, **kwargs) -> np.ndarray:
    """ Creates a MODIS-like QA raster from LANDSAT 'pixel_qa' and 'sr_aerosol' bands,
    assigns a QA rank to each pixel

    :param array pixel_qa: Landsat's pixel_qa raster as masked array
    :param array sr_aerosol: Landsat's sr_aerosol raster as masked array
    :param bool stats: whether or not to print the statistics of each filter
    :param bool plots: whether or not to save plot figures of ranks
    :param bool allplots: wheter or not to save plot figures of each filter
    :param str filename: the base file name used to save each figure
    :return rank_qa: masked array with pixel quality ranks
    """
    _stats = kwargs.get('stats', False)
    _plots = kwargs.get('plots', False)
    _allplots = kwargs.get('allplots', False)
    _fn = kwargs.get('filename', 'figure')

    assert pixel_qa.shape == sr_aerosol.shape, 'Shape mismatch in input arrays for QA generation'

    # Define the filters to use in the quality assessment of each pixel. Filters are applied bitwise on integer pixel values
    # (internally converted/operated as binary values). Filters are AND and bitwise-SHIFT operations over bit values.

    # Filters for 'pixel_qa' band (16-bit unsigned integer data type)
    clear  = 1
    water  = 2
    shadow = 3
    snow   = 4
    cloud  = 5
    cloud1 = 6   # For cirrus and cloud confidence
    cloud2 = 7   # a combination of two bits is used:
    cirrus1 = 8  #  00=not set, 01= low
    cirrus2 = 9  #  10=medium, 11=high
    NOTSET = 0  # in binary 0b0 = 0 decimal, for 'Not Set' and 'Climatology'
    LOW = 1     # in binary 0b01 = 1 decimal
    MEDIUM = 2  # in binary 0b10 = 2 decimal
    HIGH = 3    # in binary 0b11 = 3 decimal

    # Filters for the 'sr_aerosol' band (8-bit unsigned integer data type)
    fill_a = 0
    valid_a = 1
    water_a = 2
    cirrus_a = 3
    shadow_a = 4
    center_a = 5
    clilow_a = 6  # 00=Climatology, 01=Low aerosol
    medhigh_a = 7  # 10=Medium, 11=High aerosol

    # Apply the filters to create the QA Rank
    # Create the raster to hold the filter
    rank_qa = np.ones(pixel_qa.shape, dtype=pixel_qa.dtype)
    qa_mask = np.ma.getmask(pixel_qa)
    rank_qa = np.ma.masked_array(rank_qa, mask=qa_mask)
    # rank_qa *= -1

    # Identify rank 3
    water_arr = (pixel_qa >> water) & 1
    snow_arr = (pixel_qa >> snow) & 1

    rank_3 = water_arr + snow_arr
    rank_3 = rank_3.astype(bool)  # All values > 0 are rank 3

    # Identify rank 2
    clear_arr = np.logical_not((pixel_qa >> clear) & 1)  # Not clear pixels
    shadow_arr = (pixel_qa >> shadow) & 1
    cloud_arr = (pixel_qa >> cloud) & 1
    cirrus_arr = (sr_aerosol >> cirrus_a) & 1
    # Cirrus confidence
    cirrus_confidence = ((pixel_qa >> cirrus1) & 1) + (((pixel_qa >> cirrus2) & 1) << 1)
    cirrus_med = np.equal(cirrus_confidence, MEDIUM)
    cirrus_high = np.equal(cirrus_confidence, HIGH)
    # Cloud confidence
    cloud_confidence = ((pixel_qa >> cloud1) & 1) + (((pixel_qa >> cloud2) & 1) << 1)
    cloud_med = np.equal(cloud_confidence, MEDIUM)
    cloud_high = np.equal(cloud_confidence, HIGH)

    # Identify Rank 1 and Rank 0
    # Obtain level of aerosol, get and combine the values from bits 6 and 7
    aerosol_lvl = ((sr_aerosol >> clilow_a) & 1) + (((sr_aerosol >> medhigh_a) & 1) << 1)
    aerosol_clim = np.equal(aerosol_lvl, NOTSET)
    aerosol_high = np.equal(aerosol_lvl, HIGH)

    rank_0 = np.equal(aerosol_lvl, LOW)
    rank_1 = np.equal(aerosol_lvl, MEDIUM)

    rank_2 = clear_arr + shadow_arr + cirrus_med + cirrus_high + cirrus_arr + cloud_arr + cloud_med + cloud_high + aerosol_high + aerosol_clim
    rank_2 = rank_2.astype(bool)  # All values > 0 are rank 2

    # Aggregate the rank arrays into a single array
    rank_qa[np.equal(rank_0, 1)] = 0
    rank_qa[np.equal(rank_1, 1)] = 1
    rank_qa[np.equal(rank_2, 1)] = 2
    rank_qa[np.equal(rank_3, 1)] = 3
    
    # Inserting data into a masked array removes its mask, reset it
    # TODO: Investigate why the mask from 'pixel_qa' gets deleted at this point, 'sr_aerosol' mask should be used instead
    rank_qa = np.ma.masked_array(rank_qa, mask=np.ma.getmask(sr_aerosol))

    if _stats:
        array_stats('Water ', water_arr)
        array_stats('Snow  ', snow_arr)
        array_stats('Clear ', clear_arr)
        array_stats('Shadow', shadow_arr)
        array_stats('Cloud ', cloud_arr)
        array_stats('Cirrus', cirrus_arr)
        array_stats('Cirrus conf ', cirrus_confidence)
        array_stats('Cirrus med  ', cirrus_med)
        array_stats('Cirrus high ', cirrus_high)
        array_stats('Cloud conf  ', cloud_confidence)
        array_stats('Cloud med   ', cloud_med)
        array_stats('Cloud high  ', cloud_high)

        array_stats('Aerosol     ', aerosol_lvl)
        array_stats('Aerosol clim', aerosol_clim)
        array_stats('Aerosol high', aerosol_high)
        array_stats('Rank 0', rank_0)
        array_stats('Rank 1', rank_1)
        array_stats('Rank 2', rank_2)
        array_stats('Rank 3', rank_3)

    if _allplots:
        print('  --Generating intermediate plots')
        save_plot_array(clear_arr, _fn, title='clear (pixel_qa)', suffix='_pixel_qa_clear')
        save_plot_array(water_arr, _fn, title='water (pixel_qa)', suffix='_pixel_qa_water')
        save_plot_array(shadow_arr, _fn, title='shadow (pixel_qa)', suffix='_pixel_qa_shadow')
        save_plot_array(snow_arr, _fn, title='snow (pixel_qa)', suffix='_pixel_qa_snow')
        save_plot_array(cloud_arr, _fn, title='cloud (pixel_qa)', suffix='_pixel_qa_cloud')

        save_plot_array(cirrus_confidence, _fn, title='cirrus_confidence', suffix='_cirrus_confidence')
        save_plot_array(cirrus_med, _fn, title='cirrus_med', suffix='_cirrus_med')
        save_plot_array(cirrus_high, _fn, title='cirrus_high', suffix='_cirrus_high')
        save_plot_array(cloud_confidence, _fn, title='cloud_confidence', suffix='_cloud_confidence')
        save_plot_array(cloud_med, _fn, title='cloud_med', suffix='_cloud_med')
        save_plot_array(cloud_high, _fn, title='cloud_high', suffix='_cloud_high')

        save_plot_array(aerosol_clim, _fn, title='Aerosol Climatology', suffix='_sr_aerosol_clim')
        save_plot_array(aerosol_clim, _fn, title='Aerosol High', suffix='_sr_aerosol_high')

    if _plots:
        # Save the Rank plots
        print('  --Generating rank plots')
        save_plot_array(rank_0, _fn, title='Rank 0', suffix='_rank0')
        save_plot_array(rank_1, _fn, title='Rank 1', suffix='_rank1')
        save_plot_array(rank_2, _fn, title='Rank 2', suffix='_rank2')
        save_plot_array(rank_3, _fn, title='Rank 3', suffix='_rank3')

    return rank_qa


def plot_land_cover_hbar(x: list, y: list, fname: str, **kwargs) -> None:
    """ Create a horizontal bar plot to show the land cover """
    _title = kwargs.get('title', 'Distribution of land cover classes')
    _xlabel = kwargs.get('xlabel', 'Pixel count')
    _ylabel = kwargs.get('ylabel', 'Land cover class')
    _xlims = kwargs.get('xlims', (0,100))

    plt.figure(figsize=(8, 12), constrained_layout=True)
    pl = plt.barh(x, y)
    for bar in pl:
        value = bar.get_width()
        text = round(value, 4)
        if value > 0.01:
            text = round(value, 2)
        plt.annotate(text, xy=(value+0.1, bar.get_y()+0.25))
    plt.title(_title)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    if _xlims is not None:
        plt.xlim(_xlims)
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()


def plot_land_cover_sample_bars(x: list, y1: list, y2: list, fname: str, **kwargs) -> None:
    """ Creates a plot with the total and sampled land cover pixels per class """
    _title = kwargs.get('title', 'Distribution of land cover classes')
    _xlabel = kwargs.get('xlabel', 'Land cover class')
    _ylabel = kwargs.get('ylabel', 'Percentage (based on pixel count)')
    _xlims = kwargs.get('xlims', None)
    _width = kwargs.get('width', 0.35)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    rec1 = ax.bar(x - _width/2, y1, _width, label='Land cover')
    rec2 = ax.bar(x + _width - _width/2, y2, _width, label='Sample')
    # for bar in pl:
    #     value = bar.get_width()
    #     text = round(value, 4)
    #     if value > 0.01:
    #         text = round(value, 2)
    #     plt.annotate(text, xy=(value+0.1, bar.get_y()+0.25))
    plt.title(_title)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    if _xlims is not None:
        plt.xlim(_xlims)
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()


def read_keys(fn_table: str, indices: Tuple) -> Dict:
    """ Reads a table to assing numeric keys to single land cover classes and
        to its corresponding group.

    :param str fn_table: file name of the attribute table, tab delimited exported from shapefile or raster
    :param tuple indices: column numbers for land cover, land cover key, and group (GAP or INEGI)
    :return dict: LC_KEY is the key, and value is a list with [DESCRIPTIO, GRP_KEY]
    """
    # Unzip the column indexes from tuple
    # This should be: LC_KEY, DESCRIPTIO, and GRP_KEY columns in the file
    lckey, desc, grpkey, _ = indices

    # Create a dictionary with lac values and their land cover names
    land_cover_classes = {}
    with open(fn_table, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            # Skip blank lines
            if len(row) == 0:
                continue

            lc_key = int(row[lckey])
            lc_desc = row[desc]
            lc_grp = row[grpkey]

            land_cover_classes[lc_key] = [lc_desc, lc_grp]

    return land_cover_classes


def read_keys_grp(fn_table: str, indices: Tuple) -> Dict:
    # Unzip the column indexes from tuple
    # This should be: LC_KEY, DESCRIPTIO, and GRP_KEY columns in the file
    lckey, _, grpkey, desc = indices

    # Create a dictionary with lac values and their land cover names
    land_cover_groups = {}
    with open(fn_table, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            # Skip blank lines
            if len(row) == 0:
                continue

            # lc_key = int(row[lckey])
            lc_desc = row[desc]
            lc_grp = int(row[grpkey])

            land_cover_groups[lc_grp] = lc_desc

    return land_cover_groups

def land_cover_freq(fn_raster: str, fn_keys: str, **kwargs) -> Dict:
    """ Generates a single dictionary of land cover classes/groups and its pixel frequency """
    _verbose = kwargs.get('verbose', False)

    # Open the land cover raster and retrive the land cover classes
    raster_arr, nodata, metadata, geotransform, projection, epsg = open_raster(fn_raster)
    if _verbose:
        print(f'  --Opening raster: {fn_raster}')
        print(f'  ----Metadata      : {metadata}')
        print(f'  ----NoData        : {nodata}')
        print(f'  ----Columns       : {raster_arr.shape[1]}')
        print(f'  ----Rows          : {raster_arr.shape[0]}')
        print(f'  ----Geotransform  : {geotransform}')
        print(f'  ----Projection    : {projection}')
        print(f'  ----EPSG          : {epsg}')

    # First get the land cover keys in the array, then get their corresponding description
    raster_arr = raster_arr.astype(int)
    keys, freqs = np.unique(raster_arr, return_counts=True)

    if _verbose:
        print(f'  --{keys}')
        print(f'  --{len(keys)} unique land cover classes/groups in ROI.')

    land_cover_dict = {} 
    for key, freq in zip(keys, freqs):

        if type(key) is np.ma.core.MaskedConstant:
            if _verbose:
                print(f'  --Skip the MaskedConstant object: {key}')
            continue
        land_cover_dict[key] = freq

    return land_cover_dict


def land_cover_freq_wgroups(fn_raster: str, fn_keys: str, **kwargs) -> Tuple[Dict, Dict]:
    """ Generates two dictionaries of land cover classes and groups as key and its pixel frequency """
    _verbose = kwargs.get('verbose', False)
    _indices = kwargs.get('indices', (0,1,2,3))
    # _group = kwargs.get('groups', False)

    # # Get land cover keys, description and group
    land_cover_classes = read_keys(fn_keys, _indices)
    # land_cover_grp = read_keys_grp
    # unique_classes = list(land_cover_classes.keys())
    
    # if _verbose:
    #     print(f'  --Unique land cover classes: {unique_classes}')
    #     print(f'  --{len(unique_classes)} unique land cover classses.')

    # Open the land cover raster and retrive the land cover classes
    raster_arr, nodata, metadata, geotransform, projection, epsg = open_raster(fn_raster)
    if _verbose:
        print(f'  --Opening raster: {fn_raster}')
        print(f'  --Metadata      : {metadata}')
        print(f'  --NoData        : {nodata}')
        print(f'  --Columns       : {raster_arr.shape[1]}')
        print(f'  --Rows          : {raster_arr.shape[0]}')
        print(f'  --Geotransform  : {geotransform}')
        print(f'  --Projection    : {projection}')
        print(f'  --EPSG          : {epsg}')

    # First get the land cover keys in the array, then get their corresponding description
    raster_arr = raster_arr.astype(int)
    lc_keys_arr, lc_frq = np.unique(raster_arr, return_counts=True)

    if _verbose:
        print(f'  --{lc_keys_arr}')
        print(f'  --{len(lc_keys_arr)} unique land cover values in ROI.')

    land_cover_dict = {}  # a dict with, lc_key: lc_freq
    land_cover_groups_dict = {}  # a dict with cumulative frequencies per group, lc_grp: grp_freq
    for lc_key, freq in zip(lc_keys_arr, lc_frq):
        if type(lc_key) is np.ma.core.MaskedConstant:
            if _verbose:
                print(f'  --Skip the MaskedConstant object: {lc_key}')
            continue
        # if lc_key not in unique_classes:
        #     if _verbose:
        #         print(f'  Skip the MaskedConstant object: {lc_key}')
        #     continue

        # Retrieve land cover description and its group
        lc_desc = land_cover_classes[lc_key][0]
        lc_grp = land_cover_classes[lc_key][1]
        
        if _verbose:
            print(f'  --KEY={lc_key:>3} [FREQ={freq:>10}]: {lc_desc:>75} GROUP={lc_grp:<75} ', end='')
        # Save frequencies per land cover class
        land_cover_dict[lc_key] = freq

        # Accumulate frequencies per group
        if land_cover_groups_dict.get(lc_grp) is None:
            if _verbose:
                print(f'NEW group.')
            land_cover_groups_dict[lc_grp] = freq
        else:
            land_cover_groups_dict[lc_grp] += freq
            if _verbose:
                print(f'EXISTING group.')
    return land_cover_dict, land_cover_groups_dict


def land_cover_by_group(fn_raster: str, fn_keys: str, **kwargs) -> Dict:
    """ Groups the land cover classes by its group

    :param str fn_raster: file name of raster with the land cover classes
    :param str fn_keys: file to match classes with its group
    :param tuple indices: column indices with key, description, and group for file above
    :return lc_by_grp: a dict,each key (group) contains a list of its land cover classes
    """
    _verbose = kwargs.get('verbose', False)
    _fn_grp_keys = kwargs.get('fn_grp_keys', '')
    _indices = kwargs.get('indices', (0,1,2,3))

    # Get land cover keys, description and group
    # If a one-to-one file used, default indices are 0,1,2 else use custom
    land_cover_classes = read_keys(fn_keys, _indices)
    unique_classes = list(land_cover_classes.keys())
    
    if _verbose:
        print(f'  --Done. {len(unique_classes)} unique land cover classses read.')

    # Open the land cover raster and retrive the land cover classes
    raster_arr, nodata, metadata, geotransform, projection, epsg = open_raster(fn_raster)
    if _verbose:
        print(f'  --Opening raster: {fn_raster}')
        print(f'  --Metadata      : {metadata}')
        print(f'  --NoData        : {nodata}')
        print(f'  --Columns       : {raster_arr.shape[1]}')
        print(f'  --Rows          : {raster_arr.shape[0]}')
        print(f'  --Geotransform  : {geotransform}')
        print(f'  --Projection    : {projection}')
        print(f'  --EPSG          : {epsg}')

    # First get the land cover keys in the array, then get their corresponding description
    raster_arr = raster_arr.astype(int)
    lc_keys_arr = np.unique(raster_arr)

    if _verbose:
        print(f'  --{lc_keys_arr}')
        print(f'  --{len(lc_keys_arr)} unique land cover values in ROI.')
    
    lc_by_grp = {}
    for lc_key in lc_keys_arr:
        # Skip the MaskedConstant objects
        if lc_key not in unique_classes:
            if _verbose:
                print(f'  --Skip the MaskedConstant object: {lc_key}')
            continue
        # Retrieve land cover description and its group
        lc_grp = land_cover_classes[lc_key][1]

        # Save a list of land cover classes contained in each group
        if lc_by_grp.get(lc_grp) is None:
            lc_by_grp[lc_grp] = [lc_key]
        else:
            lc_by_grp[lc_grp].append(lc_key)
    
    # Optional: save to a CSV
    print('  --Saving the group keys...')
    # WARNING! Windows needs "newline=''" or it will write \r\r\n which writes an empty line between rows
    if _fn_grp_keys != '':
        with open(_fn_grp_keys, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Group', 'Land Cover Classes'])

            for i, grp in enumerate(sorted(list(lc_by_grp.keys()))):
                if _verbose:
                    print(f'  --Group key: {grp:>3}, Classes: {lc_by_grp[grp]}')
                writer.writerow([grp, ','.join(str(x) for x in lc_by_grp[grp])])

    return lc_by_grp


def reclassify_by_group(fn_raster: str, dict_grp: Dict, fn_out_raster: str, **kwargs) -> None:
    """ Reclassify a raster of land cover classes by its group
    
    :param str fn_raster: input raster with land cover classes
    :param dict dict_grp: a dict with groups of land cover classes
    """
    _verbose = kwargs.get('verbose', False)

    print('\n  --Reclassifying rasters to use land cover groups...')
    
    # Open the land cover raster and retrive the land cover classes
    raster_arr, nodata, metadata, geotransform, projection, epsg = open_raster(fn_raster)
    if _verbose:
        print(f'  --Opening raster: {fn_raster}')
        print(f'  --Metadata      : {metadata}')
        print(f'  --NoData        : {nodata}')
        print(f'  --Columns       : {raster_arr.shape[1]}')
        print(f'  --Rows          : {raster_arr.shape[0]}')
        print(f'  --Geotransform  : {geotransform}')
        print(f'  --Projection    : {projection}')
        print(f'  --EPSG          : {epsg}')

    raster_groups = np.zeros(raster_arr.shape, dtype=np.int64)

    # Use groups of land cover classes in the dict to reclassify the raster
    for i, grp in enumerate(sorted(list(dict_grp.keys()))):
        if _verbose:
            print(f'  --Group key: {grp:>3}, LC classes: {dict_grp[grp]}')
        raster_to_replace = np.zeros(raster_arr.shape, dtype=np.int64)

        # Join all the land cover classes of the same group
        for land_cover_class in dict_grp[grp]:
            raster_to_replace[np.equal(raster_arr, land_cover_class)] = grp
            raster_groups[np.equal(raster_arr, land_cover_class)] = grp
            if _verbose:
                print(f'  --Replacing {land_cover_class} with {grp}')

    if _verbose:
        print(f'  --Creating raster for groups {fn_out_raster} ...')
    create_raster(fn_out_raster, raster_groups, int(epsg), geotransform)
    print('  Reclassifying rasters to use land cover groups... done!')


# def land_cover_percentages(raster_fn: str, fn_keys: str, stats_fn: str, **kwargs) -> tuple:   
#     """ Calculate the land cover percentages from a raster_fn file

#     :param str raster_fn: name of the raster file (GeoTIFF) with the land cover classes
#     :param str fn_keys: name of a tab delimited text file that links raster keys (numeric) and land cover classes
#     :param str stats_fn: name to save a file with statistics (CSV)
#     :param tuple indices: column numbers for land cover column, land cover key column, and group column (GAP or INEGI)
#     """
#     _indices = kwargs.get('indices', (0,18,16))  # default values are for GAP/LANDCOVER attibutes text file
#     _verbose = kwargs.get('verbose', False)

#     print(f'\nCalculating land cover percentages...')

#     # Unzip the column indexes from tuple
#     col_key, col_val, col_grp = _indices

#     # Create a dictionary with values and their land cover ecosystem names
#     # land_cover_classes = {}
#     # with open(fn_keys, 'r') as csvfile:
#     #     reader = csv.reader(csvfile, delimiter='\t')
#     #     header = next(reader)
#     #     if _verbose:
#     #         print(f'  Header: {",".join(header)}')
#     #     for row in reader:
#     #         # Skip blank lines
#     #         if len(row) == 0:
#     #             continue

#     #         key = int(row[col_key])
#     #         val = row[col_val]
#     #         grp = row[col_grp]

#     #         # Too much to show
#     #         # if _verbose:
#     #         #     print(f'  {key}: {val}')
            
#     #         # land_cover_classes[key] = val
#     #         land_cover_classes[key] = [val, grp]
#     land_cover_classes = read_keys(fn_keys, _indices)
#     unique_classes = list(land_cover_classes.keys())
    
#     if _verbose:
#         print(f'  Done. {len(unique_classes)} unique land cover classses read.')

#     # Open the land cover raster and retrive the land cover classes
#     raster_arr, nodata, metadata, geotransform, projection = open_raster(raster_fn)
#     if _verbose:
#         print(f'  Opening raster: {raster_fn}')
#         print(f'  Metadata      : {metadata}')
#         print(f'  NoData        : {nodata}')
#         print(f'  Columns       : {raster_arr.shape[1]}')
#         print(f'  Rows          : {raster_arr.shape[0]}')
#         print(f'  Geotransform  : {geotransform}')
#         print(f'  Projection    : {projection}')

#     # First get the land cover keys in the array, then get their corresponding description
#     lc_keys_arr, lc_frq = np.unique(raster_arr, return_counts=True)

#     if _verbose:
#         print(f'  {lc_keys_arr}')
#         print(f'  {len(lc_keys_arr)} unique land cover values in ROI.')

#     land_cover = {}  # a dict with, lc_freq: [lc_key, description, group]
#     land_cover_groups = {}  # a dict with cumulative frequencies per group, lc_grp: grp_freq
#     for lc_key, freq in zip(lc_keys_arr, lc_frq):
#         # Skip the MaskedConstant objects
#         if lc_key not in unique_classes:
#             if _verbose:
#                 print(f'  Skip the MaskedConstant object: {lc_key}')
#             continue
#         # Retrieve land cover description and its group
#         lc_desc = land_cover_classes[lc_key][0]
#         lc_grp = land_cover_classes[lc_key][1]
        
#         if _verbose:
#             print(f'  KEY={lc_key:>3} [FREQ={freq:>10}]: {lc_desc:>75} GROUP={lc_grp:<75} ', end='')
#         land_cover[freq] = [lc_key, lc_desc, lc_grp]

#         if land_cover_groups.get(lc_grp) is None:
#             if _verbose:
#                 print(f'NEW group.')
#             land_cover_groups[lc_grp] = freq
#         else:
#             land_cover_groups[lc_grp] += freq
#             if _verbose:
#                 print(f'EXISTING group.')

#     # Calculate percentage based on pixel count of each land cover
#     counts = sorted(list(land_cover.keys()))
#     total = sum(counts)
#     percentages = (counts / total) * 100.

#     # Create lists of land cover key, its description, group, and pixel frequency
#     lc_keys = []
#     lc_description = []
#     lc_group = []
#     lc_frequency = []
#     for key_counts in counts:
#         lc_keys.append(land_cover[key_counts][0])
#         lc_description.append(land_cover[key_counts][1])
#         lc_group.append(land_cover[key_counts][2])
#         lc_frequency.append(key_counts)
#     # print(lc_group)
#     # print(lc_description)

#     # Save a file with statistics
#     print('  Saving land cover statistics file...')
#     # WARNING! Windows needs "newline=''" or it will write \r\r\n which writes an empty line between rows
#     with open(stats_fn, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         writer.writerow(['Key', 'Description', 'Group', 'Frequency', 'Percentage'])
#         for i in range(len(counts)):
#             # print(f'{lc_description[i]}: {lc_frequency[i]}, {percentages[i]}')
#             # Write each line with the land cover key, its description, group, pixel frequency, and percentage cover
#             writer.writerow([int(lc_keys[i]), lc_description[i], lc_group[i], lc_frequency[i], percentages[i]])
#     print(f'Calculating land cover percentages... done!')

#     return lc_description, percentages, land_cover_groups, raster_arr


# def land_cover_percentages_grp(land_cover_groups: dict, threshold: int = 1000, **kwargs) -> tuple:
#     """ Calculate land cover percentages by group
#     :param dict land_cover_groups:
#     :param int threshold: minimum pixel count for a land cover class to be considered in a group
#     """
#     _verbose = kwargs.get('verbose', False)

#     print('\nCalculating land cover percentages per group...')

#     # Now flip the groups dictionary to use frequency as key, and group as value
#     key_grps = list(land_cover_groups.keys())
#     lc_grps_by_freq = {}
#     for grp in key_grps:
#         lc_grps_by_freq[land_cover_groups[grp]] = grp

#     # Create lists
#     grp_filter = []
#     frq_lc = []
#     if _verbose:
#         print(f'  Removing classes with pixel count less than {threshold}')
#     grp_key_freq = sorted(list(lc_grps_by_freq.keys()))
#     for freq in grp_key_freq:
#         if freq >= threshold:
#             grp_filter.append(lc_grps_by_freq[freq])
#             frq_lc.append(freq)
#         else:
#             if _verbose:
#                 print(f'  Group "{lc_grps_by_freq[freq]}" removed by small pixel count: {freq}')
    
#     if _verbose:
#         print(f'  {len(grp_filter)} land cover groups added.')

#     # Calculate percentage based on pixel count of each land cover group
#     percent_grp = (frq_lc / sum(frq_lc)) * 100.
#     print('Calculating land cover percentages per group... done!')

#     return grp_filter, percent_grp


# def reclassify_land_cover_by_group(raster_arr: np.ndarray, raster_geotransform: list, raster_proj: int, grp_filter: list, fn_lc_stats: str, fn_grp_keys: str, fn_grp_landcover: str, **kwargs) -> None:
#     """ Creates a reclassified land cover raster using groups (groups of land cover)

#     :param str intermediate: base name for intermediate rasters, will create one per class if non empty
#     """
#     _intermediate = kwargs.get('intermediate', '')
#     _verbose = kwargs.get('verbose', False)

#     print('\nReclassifying rasters to use land cover groups...')
    
#     # Creating reclassification key
#     ecos_by_group = {}
#     with open(fn_lc_stats, 'r') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         header = next(reader)
#         for row in reader:
#             key = int(row[0])  # Keys are ecosystem, a numeric value
#             grp = row[2]  # Groups, also string

#             # Use the groups filter created before, in order to
#             # discard the groups with lower pixel count
#             if grp not in grp_filter:
#                 continue
            
#             if ecos_by_group.get(grp) is None:
#                 # Create new group
#                 ecos_by_group[grp] = [key]
#             else:
#                 # Add the ecosystem key to the group
#                 ecos_by_group[grp].append(key)

#     raster_groups = np.zeros(raster_arr.shape, dtype=np.int64)

#     print('  Saving the group keys...')
#     # WARNING! Windows needs "newline=''" or it will write \r\r\n which writes an empty line between rows
#     with open(fn_grp_keys, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         writer.writerow(['Group Key', 'Description', 'Ecosystems'])

#         for i, grp in enumerate(sorted(list(ecos_by_group.keys()))):
#             group = i+1
#             if _verbose:
#                 print(f'  Group key: {group:>3}, Description: {grp}, Classes|Ecosystems: {ecos_by_group[grp]}')
#             raster_to_replace = np.zeros(raster_arr.shape, dtype=np.int64)

#             writer.writerow([group, grp, ','.join(str(x) for x in ecos_by_group[grp])])
            
#             # Join all the ecosystems of the same group
#             for ecosystem in ecos_by_group[grp]:
#                 raster_to_replace[np.equal(raster_arr, ecosystem)] = group
#                 raster_groups[np.equal(raster_arr, ecosystem)] = group
#                 if _verbose:
#                     print(f'  --Replacing {ecosystem} with {group}')

#             # WARNING! THIS BLOCK WILL CREATE A RASTER FILE PER LAND COVER CLASS
#             if _intermediate != '':
#                 # If base name given, create intermediate rasters
#                 group_str = str(i+1).zfill(3)
#                 fn_interm_raster = f'{_intermediate}_{group_str}.tif'
#                 if _verbose:
#                     print(f'  Creating raster for group {group} in {fn_interm_raster} ...')
#                 create_raster(fn_interm_raster, raster_to_replace, raster_proj, raster_geotransform)

#     if _verbose:
#         print(f'  Creating raster for groups {fn_grp_landcover} ...')
#     create_raster(fn_grp_landcover, raster_groups, raster_proj, raster_geotransform)
#     print('Reclassifying rasters to use land cover groups... done!')


def read_params(filename: str) -> Dict:
    """ Reads the parameters from a CSV file """
    params = {}
    with open(filename, 'r') as csv_file:
        writer = csv.reader(csv_file, delimiter='=')
        for row in writer:
            if len(row) == 0:
                continue
            params[row[0]]=row[1]
    return params


def read_clr(filename: str, zero: bool=False) -> ListedColormap:
    """ Reads a colormap from a CLR file """
    _max = 255
    mycolors = []
    if zero:
        mycolors.append([255, 255, 255, 255])
        # mycolors.append([0, 0, 0, 255])
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 0:
                continue
            row_colors = [float(row[1])/_max, float(row[2])/_max, float(row[3])/_max, float(row[4])/_max]
            mycolors.append(row_colors)
    cmaplc = ListedColormap(mycolors)
    return cmaplc


# DO NOT USE THIS
# def plot_land_cover(data_arr: np.ndarray, _fncmap: str, **kwargs):
#     """ Plots a land cover array using a discrete colorbar """
#     _title = kwargs.get('title', '')
#     _title = kwargs.get('title', '')
#     _savefig = kwargs.get('savefig', '')
#     _dpi = kwargs.get('dpi', 300)
#     _vmax = kwargs.get('vmax', None)
#     _vmin = kwargs.get('vmin', None)
#     _zero = kwargs.get('zero', False)

#     # Read a custom colormap
#     # if _fncmap != '':
#     _cmap = read_clr(_fncmap, _zero)
#     bounds = [x for x in range(len(_cmap.colors))]
#     # print(f'  n_clases={_n_classes} colors={len(_cmap.colors)}')
#     print(f'  bounds={bounds}')
#     norm = matplotlib.colors.BoundaryNorm(bounds, _cmap.N)

#     fig = plt.figure()
#     fig.set_figheight(16)
#     fig.set_figwidth(12)

#     ax = plt.gca()
#     im = ax.imshow(data_arr, cmap=_cmap, vmax=_vmax, vmin=_vmin)

#     # create an axes on the right side of ax. The width of cax will be 5%
#     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     ax.grid(False)
    
#     # plt.colorbar(im, cax=cax)
#     plt.colorbar(matplotlib.cm.ScalarMappable(cmap=_cmap, norm=norm), ticks=bounds, cax=cax)

#     if _title != '':
#         plt.title(_title)
#     if _savefig != '':
#         fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)

#     # plt.show()
#     plt.close()


def plot_dataset(array: np.ndarray, **kwargs) -> None:
    """ Plots a dataset with a continuous colorbar """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    # Set max and min
    if _vmax is None and _vmin is None:
        _vmax = np.max(array)
        _vmin = np.min(array)

    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figwidth(12)

    ax = plt.gca()
    im = ax.imshow(array, cmap='jet', vmax=_vmax, vmin=_vmin)
        
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.grid(False)
    
    plt.colorbar(im, cax=cax)

    if _title != '':
        # plt.suptitle(_title)
        ax.set_title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_array_cmap(array: np.ndarray, colors_dict: dict, labels: list, **kwargs) -> None:
    """ Plots an array with unique values and labels in colormap """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)

    assert len(colors_dict) == len(labels), "Mismatch in colors and labels lists"

    # Create color map from list of colors
    cmap = ListedColormap([colors_dict[x] for x in colors_dict.keys()])
    len_lab = len(labels)

    # Prepare bins for normalizer
    norm_bins = np.sort([*colors_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # print(f'norm_bins={norm_bins}')

    # Make normalizer and formatter: puts labels every half unit
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot figure
    fig,ax = plt.subplots(figsize=(16,12))
    im = ax.imshow(array, interpolation='none', resample=False, cmap=cmap, norm=norm)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2  # puts ticks every unit
    cb = fig.colorbar(im, format=fmt, ticks=tickz)
    if _title != '':
        plt.title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    # plt.show()
    plt.close()


def plot_array_clr(array: np.ndarray, fn_clr: str, **kwargs) -> None:
    """ Plots an array using a colormap from a CLR file """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _zero = kwargs.get('zero', False)

    # Read labels
    labels = []
    if _zero:
        labels.append('0: No Data')
    with open(fn_clr, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='-')
        for row in reader:
            if len(row) == 0:
                continue
            num = row[0].split(' ')[-2]  # extract the numeric key for landcover
            # print(num)
            labels.append(num.strip() + ': ' + row[1].strip())
    # print(labels)

    mycmap = read_clr(fn_clr, zero=_zero)

    # Create a dictionary with numeric labels and colors
    colors = {}
    for k, v in enumerate(mycmap.colors):
        colors[k] = v
    # print(f'  colors={colors}')
    plot_array_cmap(array, colors, labels, title=_title, savefig=_savefig, dpi=_dpi)


def plot_hdf_dataset(filename, ds, **kwargs):
    """ Opens a HDF5 dataset and plots its data
    :param str filename: absolute path of the HDF5 file
    :param str ds: the dataset (e.g. NDVI AVG)
    """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

    ds_arr = read_from_hdf(filename, ds)

    plot_dataset(ds_arr, title=_title, savefig=_savefig, vmax=_vmax, vmin=_vmin, dpi=_dpi)


def plot_hist(ds: np.ndarray, **kwargs) -> None:
    """ Plots a histogram of features from a dataset """
    _feature = kwargs.get('feature', '')
    _bins = kwargs.get('bins', 30)
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)

    ds = ds.flatten()
    
    fig = plt.figure(figsize=(16,12), tight_layout=True)

    plt.hist(ds, bins=_bins)  # histogram of all values

    if _title != '':
        plt.title(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_2hist(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots 2 histograms side by side. """
    _feature = kwargs.get('feature', '')
    _bins = kwargs.get('bins', 30)
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _2half = kwargs.get('half', True)  # half the bins in second histogram
    
    ds1 = ds1.flatten()
    ds2 = ds2.flatten()

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(ds1, bins=_bins)
    axs[1].hist(ds2, bins=_bins//2 if _2half else _bins)

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_monthly_bak(var: str, ds: str, cwd: str, **kwargs):
    """ Plots monthly 2D values from a HDF5 file by reading the variable and the dataset
    :param str var: the variable (e.g. NDVI)
    :param str ds: the dataset (e.g. NDVI AVG)
    :param str cwd: current working directory to open the HDF5 file
    """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, ax = plt.subplots(3, 4, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    for n, month in enumerate(months):
        # fn = cwd + f'MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf' # C1
        fn = cwd + f'MONTHLY.{var.upper()}.{month}.hdf'
        print(f'  --File name:{fn}')
        ds_arr = read_from_hdf(fn, ds)

        # Set max and min
        if _vmax is None and _vmin is None:
            _vmax = np.max(ds_arr)
            _vmin = np.min(ds_arr)

        row = n//4
        col = n%4
        im=ax[row,col].imshow(ds_arr, cmap='jet', vmax=_vmax, vmin=_vmin)
        ax[row,col].set_title(month)
        ax[row,col].axis('off')
   
    # fig.tight_layout()

    # Single colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # Single colorbar, easier
    fig.colorbar(im, ax=ax.ravel().tolist())

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.show()
    plt.close()


def plot_monthly(var: str, ds: str, cwd: str, **kwargs):
    """ Plots monthly 2D values from a HDF5 file by reading the variable and the dataset
    :param str var: the variable (e.g. NDVI)
    :param str ds: the dataset (e.g. NDVI AVG)
    :param str cwd: current working directory to open the HDF5 file
    """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    _cmap = kwargs.get('cmap', 'jet')
    _nan = kwargs.get('nan', -10000)  # NaN threshold

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, ax = plt.subplots(3, 4, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    _cmap = matplotlib.colormaps[_cmap]
    _cmap.set_bad(color='magenta')

    for n, month in enumerate(months):
        # fn = cwd + f'MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf' # C1
        fn = cwd + f'MONTHLY.{var.upper()}.{month}.hdf'
        print(f'  --File name:{fn}')
        ds_arr = read_from_hdf(fn, ds)

        # Set max and min
        if _vmax is None and _vmin is None:
            _vmax = np.max(ds_arr)
            _vmin = np.min(ds_arr)

        # Calculate the percentage of missing data
        ds_arr = np.ma.array(ds_arr, mask=(ds_arr < _nan))
        percent = (np.ma.count_masked(ds_arr)/ds_arr.size) * 100
        print(f"    --Missing: {np.ma.count_masked(ds_arr)}/{ds_arr.size}={percent:>0.2f}")

        row = n//4
        col = n%4
        im=ax[row,col].imshow(ds_arr, cmap=_cmap, vmax=_vmax, vmin=_vmin)
        ax[row,col].set_title(month + f' ({percent:>0.2f}% NaN)')
        ax[row,col].axis('off')
   
    # Single colorbar, easier
    fig.colorbar(im, ax=ax.ravel().tolist())

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_monthly_hist(var: str, ds: str, cwd: str, **kwargs) -> None:
    """ Plots monthly histograms from a HDF5 file by reading the variable and the dataset
    :param str var: the variable (e.g. NDVI)
    :param str ds: the dataset (e.g. NDVI AVG)
    :param str cwd: current working directory to open the HDF5 file
    """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _bins = kwargs.get('bins', 30)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, ax = plt.subplots(3, 4, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    for n, month in enumerate(months):
        # fn = cwd + f'02_STATS/MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf'
        # fn = cwd + f'data/landsat/C2/02_STATS/MONTHLY.{var.upper()}.{month}.hdf'
        fn = cwd + f'MONTHLY.{var.upper()}.{month}.hdf'
        print(f'  --File name:{fn}')
        ds_arr = read_from_hdf(fn, ds)

        row = n//4
        col = n%4

        ax[row,col].hist(ds_arr.flatten(), bins=_bins, color='blue')  # histogram of all values
        ax[row,col].set_title(month)

        # Leave labels on left and bottom axis only
        if col != 0:
            ax[row,col].set_yticklabels([])
            ax[row,col].tick_params(left=False)
        if row != 2:
            ax[row,col].set_xticklabels([])
            ax[row,col].tick_params(bottom=False)

    # Share the y-axis along rows
    ax[0, 0].get_shared_y_axes().join(ax[0,0], *ax[0,:])
    # ax[0, 0].autoscale()

    ax[1, 0].get_shared_y_axes().join(ax[1,0], *ax[1,:])
    ax[1, 0].autoscale()

    ax[2, 0].get_shared_y_axes().join(ax[2,0], *ax[2,:])
    ax[2, 0].autoscale()

    # Share the x-axis along columns
    ax[0, 0].get_shared_x_axes().join(ax[0,0], *ax[:,0])
    ax[0, 0].autoscale()

    ax[0, 1].get_shared_x_axes().join(ax[0,1], *ax[:,1])
    ax[0, 1].autoscale()

    ax[0, 2].get_shared_x_axes().join(ax[0,2], *ax[:,2])
    ax[0, 2].autoscale()

    ax[0, 3].get_shared_x_axes().join(ax[0,3], *ax[:,3])
    ax[0, 3].autoscale()

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def basic_stats(fn_hdf_feat, fn_hdf_lbl, fn_csv = ''):
    """ Generates basic stats from raw data (before preprocessing) """
    ls = []
    names = ['Key', 'Type', 'Variable', 'Min Raw', 'Max Raw', 'Mean Raw', 'Min', 'Max', 'Mean']

    # Check labels
    with h5py.File(fn_hdf_lbl, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            print(f"  --Analyzing {i:>3}/{len(keys):>3}:{key:>22}", end='')
            ds = f[key][:]
            print(f"{str(ds.dtype):>8}", end='')
            _min = np.nanmin(ds)
            _max = np.nanmax(ds)
            avg = np.nanmean(ds)
            u = np.unique(ds)
            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}")
            print(f"  --unique: {len(u)}: {u}")

    # Check features
    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())

        for i, key in enumerate(keys):
            row = []
            print(f"  --{i:>3}/{len(keys):>3}:{key:>22}", end='')

            row.append(key)
            ds = f[key][:]
            # print(f"{str(ds.dtype):>8}{str(ds.shape):>13}", end='')
            print(f"{str(ds.dtype):>8}", end='')
            
            minima = MIN_BAND
            # Add the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                feat_type = 'PHEN'
                minima = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                feat_type = 'VI'
                minima = MIN_VI
            print(f"{feat_type:>5}", end='')
            row.append(feat_type)
            
            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minima = MIN_PHEN
            
            var = 'VAL'
            if key[-3:] == 'AVG':
                var = 'AVG'
            elif key[-3:] == 'MAX' and feat_type != 'PHEN':
                var = 'MAX'
            elif key[-3:] == 'MIN':
                var = 'MIN'
            elif key[-3:] == 'els':
                var = 'NPI'
            elif key[-3:] == 'DEV':
                var = 'STD'
            print(f"{var:>4}", end='')
            row.append(var)

            _min = np.nanmin(ds)
            _max = np.nanmax(ds)
            avg = np.nanmean(ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}", end='')

            # Remove extreme negative values (custom NANs)
            valid_ds = np.where(ds >= minima, ds, np.nan)

            _min = np.nanmin(valid_ds)
            _max = np.nanmax(valid_ds)
            avg = np.nanmean(valid_ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}")

            ls.append(row)
            
        # Save stats to a CSV file
        if fn_csv != '':
            df = pd.DataFrame(ls)
            df.columns = names  # rename columns
            print(f'  --{df.shape}')
            print(f'  --{df.info()}')
            df.to_csv(fn_csv)
            print(f'  --Feature stats saved to: {fn_csv}.')


def plot_2hist_bands(fn_hdf_feat, fn_hist_plot):
    """ Plots histograms of all the bands in the HDF file, two plots are generated: one with all values, and a second
        plot removes the values out of the valid range."""
    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            start = datetime.now()
            ds = f[key][:]
            # print(f'  --ds={ds.shape}')

            # Remove values out of the valid range
            minima = MIN_BAND
            # Add the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                minima = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                minima = MIN_VI
            
            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minima = MIN_PHEN

            # print('  --Plotting histogram...')
            ds1 = ds.flatten()
            # print(f'  --ds1={ds1.shape}')
            ds2 = np.where(ds1 >= minima, ds1, np.nan)
            # print(f'  --ds2={ds2.shape}')
            
            plot_2hist(ds1, ds2, title=key, half=True, bins=30, savefig=fn_hist_plot + ' ' + key + '.png')
            elapsed = datetime.now() - start
            print(f'  --Plotting histogram {key:>20} in {elapsed}.')
            plt.close()


def range_of_type(feat_type: str, df: pd.DataFrame, **kwargs) -> None:
    """ Shows the value range of a type of feature
    :param str feat_type: the type of feature BAND, VI, or PHEN
    :param dataframe df: Pandas DataFrame with the statistics from a CSV file from 'basic_stats'
    """
    _verbose = kwargs.get('verbose', False)
    # Get the range for the specified type
    print(f"  --Showing range for type: {feat_type}")
    df_feats = df.loc[df['Type'] == feat_type]

    if _verbose:
        print(f'  --{df_feats.head()}')
        print(f'  --{df_feats.shape}')

    print(f"  --{'Variable':>10} {'Minima':>10} {'Maxima':>10} {'Raw Min':>10} {'Raw Max':>10} {'Sum Min':>10}")

    if feat_type == 'PHEN':
        df_pheno = df_feats.loc[df_feats['Variable'] == 'VAL']
        rows, _ = df_pheno.shape
        for i in range(rows):
            print(f"  --{df_pheno.iloc[i]['Key']:>10} {df_pheno.iloc[i]['Min']:>10.2f} {df_pheno.iloc[i]['Max']:>10.2f} {df_pheno.iloc[i]['Min Raw']:>10.2f} {df_pheno.iloc[i]['Max Raw']:>10.2f} {'--':>10}")
    else:
        
        avg = df_feats.loc[df_feats['Variable'] == 'AVG']
        if _verbose:
            print(f'  --{avg.head()}')
            print(f'  --{avg.shape}')
        print(f"  --{'AVG':>10} {np.min(avg['Min']):>10.2f} {np.max(avg['Max']):>10.2f} {np.min(avg['Min Raw']):>10.2f} {np.max(avg['Max Raw']):>10.2f} {'--':>10}")
        
        _min = df_feats.loc[df_feats['Variable'] == 'MIN']
        if _verbose:
            print(f'  --{_min.head()}')
            print(f'  --{_min.shape}')
        print(f"  --{'MIN':>10} {np.min(_min['Min']):>10.2f} {np.max(_min['Max']):>10.2f} {np.min(_min['Min Raw']):>10.2f} {np.max(_min['Max Raw']):>10.2f} {'--':>10}")

        _max = df_feats.loc[df_feats['Variable'] == 'MAX']
        if _verbose:
            print(f'  --{_max.head()}')
            print(f'  --{_max.shape}')
        print(f"  --{'MAX':>10} {np.min(_max['Min']):>10.2f} {np.max(_max['Max']):>10.2f} {np.min(_max['Min Raw']):>10.2f} {np.max(_max['Max Raw']):>10.2f} {'--':>10}")

        std = df_feats.loc[df_feats['Variable'] == 'STD']
        if _verbose:
            print(f'  --{std.head()}')
            print(f'  --{std.shape}')
        print(f"  --{'STD':>10} {np.min(std['Min']):>10.2f} {np.max(std['Max']):>10.2f} {np.min(std['Min Raw']):>10.2f} {np.max(std['Max Raw']):>10.2f} {'--':>10}")

        npixels = df_feats.loc[df_feats['Variable'] == 'NPI']
        if _verbose:
            print(f'  --{npixels.head()}')
            print(f'  --{npixels.shape}')
        print(f"  --{'NPI':>10} {np.min(npixels['Min']):>10.2f} {np.max(npixels['Max']):>10.2f} {np.min(npixels['Min Raw']):>10.2f} {np.max(npixels['Max Raw']):>10.2f} {np.sum(npixels['Min']):>10.2f}")


def normalize(ds: np.ndarray) -> np.ndarray:
    """ Normalize a dataset with min-max feature scaling into range [0,1] """
    _min = np.nanmin(ds)
    _max = np.nanmax(ds)
    return (ds - _min) / (_max - _min)


def standardize(ds: np.ndarray) -> np.ndarray:
    """ Standarize a dataset into range [-1, 1] """
    avg = np.nanmean(ds)
    std = np.nanstd(ds)
    return (ds - avg) / std


def get_files(directory: str, ext: str) -> List:
    """ Returns a list of files in the directory of the specified extension"""
    lenext = len(ext)
    abspath = os.path.abspath(directory)
    if os.path.exists(abspath):
        dir_list = os.listdir(abspath)
        onlyfiles = []
        for item in dir_list:
            absfile = os.path.join(abspath, item)
            if os.path.isfile(absfile) and absfile[-lenext:] == ext:
                onlyfiles.append(absfile)
    else:
        return None
    return onlyfiles


def plot_2histograms(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots 2 histograms side by side, same scale. """
    _bins = kwargs.get('bins', 30)
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _titles = kwargs.get('titles', ('', ''))
    _xlims = kwargs.get('xlims', (0,0))
    
    ds1 = ds1.flatten()
    ds2 = ds2.flatten()

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(12,8))

    # First histogram
    N, bins, patches = axs[0].hist(ds1, bins=_bins)
    axs[0].grid(True, linestyle='--')
    # Set title for first histogram
    if _titles[0] != '':
        axs[0].set_title(_titles[0])
    # Match the xlims to accurate comparisons
    if _xlims != (0,0):
        axs[0].set_xlim(_xlims)

    # Color first histogram by height
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # print(bins)

    # Second histogram, use the same bins as histogram 1
    N, bins, patches =  axs[1].hist(ds2, bins=bins)
    axs[1].grid(True, linestyle='--')
    # Set title for second histogram
    if _titles[1] != '':
        axs[1].set_title(_titles[1])
    # Match the xlims to accurate comparisons
    if _xlims != (0,0):
        axs[1].set_xlim(_xlims)
    
    # Color second histogram by height
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_2dataset(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots 2 datasets side by side. """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _titles = kwargs.get('titles', ('', ''))
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

    # Set max and min
    if _vmax is None and _vmin is None:
        _vmax = np.max(ds1) if np.max(ds1) > np.max(ds2) else np.max(ds2)
        _vmin = np.min(ds1) if np.min(ds1) < np.min(ds2) else np.min(ds2)
    
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12,8))

    # First plot
    im1 = axs[0].imshow(ds1, vmax=_vmax, vmin=_vmin)
    axs[0].grid(True, linestyle='--')
    # Set title for first plot
    if _titles[0] != '':
        axs[0].set_title(_titles[0])

    # Second plot
    im2 = axs[1].imshow(ds2, vmax=_vmax, vmin=_vmin)
    axs[1].grid(True, linestyle='--')
    # Set title for second plot
    if _titles[1] != '':
        axs[1].set_title(_titles[1])

    # Add space for colour bar
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    plt.colorbar(im2, cax=cax)

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_diff(ds1: np.ndarray, ds2: np.ndarray, ds3: np.ndarray, **kwargs) -> None:
    """ Plots 2 datasets side by side and a third plot of their difference. """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _titles = kwargs.get('titles', ('', ''))
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

        # Set max and min
    if _vmax is None and _vmin is None:
        _vmax = np.max(ds1) if np.max(ds1) > np.max(ds2) else np.max(ds2)
        _vmin = np.min(ds1) if np.min(ds1) < np.min(ds2) else np.min(ds2)
    # print(f"VMax: {_vmax} VMin: {_vmin}")
    
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(24,8))

    # First plot
    im1 = axs[0].imshow(ds1, vmax=_vmax, vmin=_vmin)
    axs[0].grid(True, linestyle='--')
    # Set title for first plot
    if _titles[0] != '':
        axs[0].set_title(_titles[0])
    plt.colorbar(im1, ax=axs[0])

    # Second plot
    im2 = axs[1].imshow(ds2, vmax=_vmax, vmin=_vmin)
    axs[1].grid(True, linestyle='--')
    # Set title for second plot
    if _titles[1] != '':
        axs[1].set_title(_titles[1])
    plt.colorbar(im2, ax=axs[1])
    
    # Third plot, difference
    _vmax = np.nanmax(ds3)
    _vmin = np.nanmin(ds3)
    im3 = axs[2].imshow(ds3, vmax=_vmax, vmin=_vmin)
    axs[2].grid(True, linestyle='--')
    # Set title for third plot
    if _titles[2] != '':
        axs[2].set_title(_titles[2])
    plt.colorbar(im3, ax=axs[2])

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_identity(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots a scatterplot with a 1:1 line between two variables and 
        their linear model obtained with OLS from 'statsmodels' module.
        This is slow when datasets have a HUGE amount of points.
    """
    _title = kwargs.get('title', '')
    _xlabel = kwargs.get('xlabel', '')
    _ylabel = kwargs.get('ylabel', '')
    _savefig = kwargs.get('savefig', '')
    _lims = kwargs.get('lims', (0,0))
    _model = kwargs.get('model', None)  # OLS results from statsmodels
    _dpi = kwargs.get('dpi', 300)

    # ds1 = ds1.flatten()
    # ds2 = ds2.flatten()

    fig = plt.figure()

    x, y = np.linspace(0,10000,10001), np.linspace(0,10000,10001)

    plt.scatter(ds1, ds2, marker='.', label='Data', s=1)
    plt.plot(x, y, '-k', label='1:1')

    if _model is not None:
        # print(_params.const, _params.DS1)
        # plt.plot(x, _params.const + x * _params.DS1, color='green', linestyle='dashed')
        # No constant (intercept), assuming it's zero
        plt.plot(x, x * _model.params.DS1, color='green', linestyle='dashed',
                 label=f"y={_model.params.DS1:>0.2f} x (R^2={_model.rsquared:>0.4f})")

    if _lims != (0,0):
        plt.xlim(_lims)
        plt.ylim(_lims)
    if _xlabel != '':
        plt.xlabel(_xlabel)
    if _ylabel != '':
        plt.ylabel(_ylabel)
    if _title != '':
        plt.title(_title)
    
    plt.axis('equal')
    plt.legend(loc='best')

    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)

    plt.close()


def plot_heatmap2d(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots a heatmap of two variables with HUGE amount of points, a straight 1:1 line,
        and their linear model obtained with OLS from the 'statsmodels' module.
        This is faster than a scatterplot.
    """
    _title = kwargs.get('title', '')
    _xlabel = kwargs.get('xlabel', '')
    _ylabel = kwargs.get('ylabel', '')
    _savefig = kwargs.get('savefig', '')
    _lims = kwargs.get('lims', (0,0))
    _model = kwargs.get('model', None)  # OLS results from statsmodels
    _dpi = kwargs.get('dpi', 300)
    _bins = kwargs.get('bins', 100)
    _log = kwargs.get('log', False)

    fig = plt.figure(figsize=(14,12))

    if _lims != (0,0):
        x, y = np.linspace(_lims[0],_lims[1],abs(_lims[1]-_lims[0])+1), np.linspace(_lims[0],_lims[1],abs(_lims[1]-_lims[0])+1)
    else:
        x, y = np.linspace(0,10000,10001), np.linspace(0,10000,10001)

    plt.clf()
    heatmap, xedges, yedges = np.histogram2d(ds1, ds2, bins=_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    if _log:
        plt.imshow(heatmap.T, extent=extent, norm=colors.LogNorm(vmin=1, vmax=heatmap.max()), origin='lower', cmap='YlOrBr') # 'gist_ncar', 'gist_stern_r'
    else:
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='gist_stern') # 'gist_ncar'

    plt.plot(x, y, '-k', label='1:1')

    if _model is not None:
        # print(_params.const, _params.DS1)
        # plt.plot(x, _params.const + x * _params.DS1, color='green', linestyle='dashed', label=f"{_params.DS1:>0.2f} x + {_params.const:>0.2f}")
        # No constant (intercept), assuming it's zero
        plt.plot(x, x * _model.params.DS1, color='green', linestyle='dashed',
                 label=f"y={_model.params.DS1:>0.2f}x ($R^2$={_model.rsquared:>0.3f})")

    if _lims != (0,0):
        plt.xlim(_lims)
        plt.ylim(_lims)
    if _xlabel != '':
        plt.xlabel(_xlabel)
    if _ylabel != '':
        plt.ylabel(_ylabel)
    if _title != '':
        plt.title(_title)
    
    # plt.axis('equal')
    plt.legend(loc='best')
    plt.colorbar()

    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)

    plt.close()


def plot_corr2(ds1: np.ndarray, ds2: np.ndarray, **kwargs) -> None:
    """ Plots a heatmap of two variables with HUGE amount of points,
        and their linear model obtained with OLS from the 'statsmodels' module.
        This is faster than a scatterplot.
    """
    _title = kwargs.get('title', '')
    _xlabel = kwargs.get('xlabel', '')
    _ylabel = kwargs.get('ylabel', '')
    _savefig = kwargs.get('savefig', '')
    _xlims = kwargs.get('xlims', (0,0))
    _ylims = kwargs.get('ylims', (0,0))
    _lims = kwargs.get('lims', (0,0))
    _model = kwargs.get('model', None)  # OLS results from statsmodels
    _dpi = kwargs.get('dpi', 300)
    _bins = kwargs.get('bins', 100)
    _log = kwargs.get('log', False)

    fig = plt.figure(figsize=(14,12))

    if _lims != (0,0):
        x = np.linspace(_lims[0], _lims[1], abs(_lims[1]-_lims[0])+1)
        y = np.linspace(_lims[0], _lims[1], abs(_lims[1]-_lims[0])+1)
    elif _xlims != (0,0) and _ylims != (0,0):
        _min = _xlims[0] if _xlims[0] < _ylims[0] else _ylims[0]
        _max = _xlims[1] if _xlims[1] > _ylims[1] else _ylims[1]
        x = np.linspace(_min, _max, abs(_max-_min)+1)
        y = np.linspace(_min, _max, abs(_max-_min)+1)
    else:
        x, y = np.linspace(0,10000,10001), np.linspace(0,10000,10001)

    plt.clf()
    heatmap, xedges, yedges = np.histogram2d(ds1, ds2, bins=_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    if _log:
        plt.imshow(heatmap.T, extent=extent, norm=colors.LogNorm(vmin=1, vmax=heatmap.max()), origin='lower', cmap='YlOrBr') # 'gist_ncar', 'gist_stern_r'
    else:
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='gist_stern') # 'gist_ncar'

    plt.plot(x, y, '-k', label='1:1')

    if _model is not None:
        # print(_params.const, _params.DS1)
        plt.plot(x, _model.params.const + x * _model.params.DS1,
                 color='green',
                 linestyle='dashed',
                 label=f"{_model.params.DS1:>0.2f}x + {_model.params.const:>0.2f} ($R^2$={_model.rsquared:>0.3f})")
        # No constant (intercept), assuming it's zero
        # plt.plot(x, x * _model.params.DS1,
        #          color='green',
        #          linestyle='dashed',
        #          label=f"y={_model.params.DS1:>0.2f}x ($R^2$={_model.rsquared:>0.3f})")

    if _lims != (0,0):
        plt.xlim(_lims)
        plt.ylim(_lims)
    elif _xlims != (0,0) and _ylims != (0,0):
        plt.xlim(_xlims)
        plt.ylim(_ylims)
    if _xlabel != '':
        plt.xlabel(_xlabel)
    if _ylabel != '':
        plt.ylabel(_ylabel)
    if _title != '':
        plt.title(_title)
    
    # plt.axis('equal')
    plt.legend(loc='best')
    if _log:
        clb = plt.colorbar()
        clb.ax.set_title('Log')
    else:
        plt.colorbar()

    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()

    plt.close()

def fill_with_mean(dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills NaNs using the mean of all valid data """
    _verbose = kwargs.get('verbose', False)
    _var = kwargs.get('var', '')

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = round(np.nanmean(valid_ds), 2)
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    if _verbose:
        print(f'  --Missing {_var} values filled with {fill_value} successfully!')
    return filled_ds


def fill_with_int_mean(dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills NaNs using the mean of all valid data """
    _verbose = kwargs.get('verbose', False)
    _var = kwargs.get('var', '')

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = int(np.nanmean(valid_ds))
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    if _verbose:
        print(f'  --Missing {_var} values filled with {fill_value} successfully!')
    return filled_ds


def fill_season(sos: np.ndarray, eos: np.ndarray, los: np.ndarray, min_value: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Fills missing values from SOS, EOS and LOS 

        NOTICE: This is a window-based method and needs to know the number of rows and columns to work properly.
        """
        # _nan = kwargs.get('nan', -1)
        _max_row = kwargs.get('max_row', None)
        _max_col = kwargs.get('max_col', None)
        _verbose = kwargs.get('verbose', False)
        # _col_pixels = kwargs.get('col_pixels', 1000)
        _row_pixels = kwargs.get('row_pixels', 1000)
        _id = kwargs.get('id', '')
        
        ### SOS
        # sos = sos.astype(int)
        sos_nan_indices = np.transpose((sos<min_value).nonzero())  # get NaN indices
        if _verbose:
            print(f'  --Missing data found at SOS: {len(sos_nan_indices)}')

        ### EOS
        # eos = eos.astype(int)
        eos_nan_indices = np.transpose((eos<min_value).nonzero())
        if _verbose:
            print(f'  --Missing data found at EOS: {len(eos_nan_indices)}')

        ### LOS
        # los = los.astype(int)
        los_nan_indices = np.transpose((los<min_value).nonzero())
        if _verbose:
            print(f'  --Missing data found at LOS: {len(eos_nan_indices)}')

        # Temporary array to contain fill values in their right position
        filled_sos = sos.copy()
        filled_eos = eos.copy()
        filled_los = los.copy()

        assert sos_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape SOS={sos_nan_indices.shape} EOS={eos_nan_indices.shape}"
        assert los_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape LOS={los_nan_indices.shape} EOS={eos_nan_indices.shape}"

        # Each NaN position contains a [row, col]
        for sos_pos, eos_pos, los_pos in zip(sos_nan_indices, eos_nan_indices, los_nan_indices):
            assert np.array_equal(sos_pos, eos_pos), f"NaN positions are different SOS={sos_pos} EOS={eos_pos}"
            assert np.array_equal(sos_pos, los_pos), f"NaN positions are different SOS={sos_pos} LOS={los_pos}"

            row, col = sos_pos
            nan_value = sos[row, col]  # current position of NaN value
            # print(nan_value)
            
            win_size = 1
            removed_success = False
            while not removed_success:
                # Window to slice around the missing value
                row_start = row-win_size
                row_end = row+win_size+1
                col_start = col-win_size
                col_end = col+win_size+1
                # Adjust row,col to use for slicing when point near the edges
                if _max_row is not None:
                    if row_start < 0:
                        row_start = 0
                    if row_end > _max_row:
                        row_end = _max_row
                if _max_col is not None:
                    if col_start < 0:
                        col_start = 0
                    if col_end > _max_col:
                        col_end = _max_col
                
                # Slice a window of values around missing value
                window_sos = sos[row_start:row_end, col_start:col_end]
                window_eos = eos[row_start:row_end, col_start:col_end]

                win_values_sos = window_sos.flatten().tolist()
                win_values_eos = window_eos.flatten().tolist()

                # Remove NaN values from the list
                all_vals_sos = win_values_sos.copy()
                # Keep all values but the one at the center of window, aka the NaN value
                win_values_sos = [i for i in all_vals_sos if i != nan_value]
                all_vals_eos = win_values_eos.copy()
                win_values_eos = [i for i in all_vals_eos if i != nan_value]

                # If list is empty, it means window had only missing values, increase window
                if len(win_values_sos) == len(win_values_eos) and len(win_values_eos) > 0:
                    # List is not empty, non NaN values found!
                    removed_success = True
                    if _verbose:
                        print(f'  -- {_id}: Success with window size {win_size}. ({row},{col})')
                    break
                # If failure, increase window size and try again
                win_size += 1
            
            # For SOS use mode (will return minimum value as default)
            fill_value_sos = stats.mode(win_values_sos, keepdims=False)[0]
            if _verbose:
                print(f'  -- Fill SOS value={fill_value_sos}')

            # For EOS use aither mode or max value
            # fill_value_eos, counts = stats.mode(win_values_eos, keepdims=False)[0]
            fill_value_eos, count = stats.mode(win_values_eos, keepdims=False)
            if fill_value_eos == np.min(win_values_eos) and count == 1:
                if _verbose:
                    print(f"  -- Fill EOS value={fill_value_eos} w/count={count} isn't a true mode, use maximum instead.")
                # If default (minimum) return maximum value
                fill_value_eos = np.max(win_values_eos)
            
            # Fill value for LOS
            fill_value_los = fill_value_eos - fill_value_sos
            if fill_value_los <= 0:
                fill_value_los = 365  # assume LOS for the entire year

            if _verbose:
                print(f'  --SOS: {row},{col}: {nan_value}, values={win_values_sos}, fill_val={fill_value_sos}')
                print(f'  --EOS: {row},{col}: {nan_value}, values={win_values_eos}, fill_val={fill_value_eos}')
                print(f'  --LOS: {row},{col}: {nan_value}, fill_val={fill_value_los}\n')
            
            # Fill the missing values in their right position
            filled_sos[row, col] = fill_value_sos
            filled_eos[row, col] = fill_value_eos
            filled_los[row, col] = fill_value_los
        
        return filled_sos, filled_eos, filled_los


def fill_season_orig(sos: np.ndarray, eos: np.ndarray, los: np.ndarray, min_value: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Fills missing values from SOS, EOS and LOS 

    NOTICE: This is a window-based method and needs to know the number of rows and columns to work properly.
    """
    # _nan = kwargs.get('nan', -1)
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    # _col_pixels = kwargs.get('col_pixels', 1000)
    _row_pixels = kwargs.get('row_pixels', 1000)
    _id = kwargs.get('id', '')
    
    ### SOS
    # sos = sos.astype(int)
    sos_nan_indices = np.transpose((sos<min_value).nonzero())  # get NaN indices
    if _verbose:
        print(f'  --Missing data found at SOS: {len(sos_nan_indices)}')

    ### EOS
    # eos = eos.astype(int)
    eos_nan_indices = np.transpose((eos<min_value).nonzero())
    if _verbose:
        print(f'  --Missing data found at EOS: {len(eos_nan_indices)}')

    ### LOS
    # los = los.astype(int)
    los_nan_indices = np.transpose((los<min_value).nonzero())
    if _verbose:
        print(f'  --Missing data found at LOS: {len(eos_nan_indices)}')

    # Temporary array to contain fill values in their right position
    filled_sos = sos.copy()
    filled_eos = eos.copy()
    filled_los = los.copy()

    assert sos_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape SOS={sos_nan_indices.shape} EOS={eos_nan_indices.shape}"
    assert los_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape LOS={los_nan_indices.shape} EOS={eos_nan_indices.shape}"

    # Each NaN position contains a [row, col]
    for sos_pos, eos_pos, los_pos in zip(sos_nan_indices, eos_nan_indices, los_nan_indices):
        assert np.array_equal(sos_pos, eos_pos), f"NaN positions are different SOS={sos_pos} EOS={eos_pos}"
        assert np.array_equal(sos_pos, los_pos), f"NaN positions are different SOS={sos_pos} LOS={los_pos}"

        row, col = sos_pos
        nan_value = sos[row, col]  # current position of NaN value
        # print(nan_value)
        
        win_size = 1
        removed_success = False
        while not removed_success:
            # Window to slice around the missing value
            row_start = row-win_size
            row_end = row+win_size+1
            col_start = col-win_size
            col_end = col+win_size+1
            # Adjust row,col to use for slicing when point near the edges
            if _max_row is not None:
                if row_start < 0:
                    row_start = 0
                if row_end > _max_row:
                    row_end = _max_row
            if _max_col is not None:
                if col_start < 0:
                    col_start = 0
                if col_end > _max_col:
                    col_end = _max_col
            
            # Slice a window of values around missing value
            window_sos = sos[row_start:row_end, col_start:col_end]
            window_eos = eos[row_start:row_end, col_start:col_end]

            win_values_sos = window_sos.flatten().tolist()
            win_values_eos = window_eos.flatten().tolist()

            # Remove NaN values from the list
            all_vals_sos = win_values_sos.copy()
            # Keep all values but the one at the center of window, aka the NaN value
            win_values_sos = [i for i in all_vals_sos if i != nan_value]
            all_vals_eos = win_values_eos.copy()
            win_values_eos = [i for i in all_vals_eos if i != nan_value]

            # If list is empty, it means window had only missing values, increase window
            if len(win_values_sos) == len(win_values_eos) and len(win_values_eos) > 0:
                # List is not empty, non NaN values found!
                removed_success = True
                if _verbose:
                    print(f'  -- {_id}: Success with window size {win_size}. ({row},{col})')
                break
            # If failure, increase window size and try again
            win_size += 1
        
        # For SOS use mode (will return minimum value as default)
        fill_value_sos = stats.mode(win_values_sos, keepdims=False)[0]
        if _verbose:
            print(f'  -- Fill SOS value={fill_value_sos}')

        # For EOS use aither mode or max value
        # fill_value_eos, counts = stats.mode(win_values_eos, keepdims=False)[0]
        fill_value_eos, count = stats.mode(win_values_eos, keepdims=False)
        if fill_value_eos == np.min(win_values_eos) and count == 1:
            if _verbose:
                print(f"  -- Fill EOS value={fill_value_eos} w/count={count} isn't a true mode, use maximum instead.")
            # If default (minimum) return maximum value
            fill_value_eos = np.max(win_values_eos)
        
        # Fill value for LOS
        fill_value_los = fill_value_eos - fill_value_sos
        if fill_value_los <= 0:
            fill_value_los = 365  # assume LOS for the entire year

        if _verbose:
            print(f'  --SOS: {row},{col}: {nan_value}, values={win_values_sos}, fill_val={fill_value_sos}')
            print(f'  --EOS: {row},{col}: {nan_value}, values={win_values_eos}, fill_val={fill_value_eos}')
            print(f'  --LOS: {row},{col}: {nan_value}, fill_val={fill_value_los}\n')
        
        # Fill the missing values in their right position
        filled_sos[row, col] = fill_value_sos
        filled_eos[row, col] = fill_value_eos
        filled_los[row, col] = fill_value_los
    
    return filled_sos, filled_eos, filled_los


def fill_with_mode(data: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills missing values with the mode from the surrounding window
    
    NOTICE: This is a window-based method and needs to know the number of rows and columns to work properly.
    """
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    _row_pixels = kwargs.get('row_pixels', 1000)
    _id = kwargs.get('id', '')
    
    data = data.astype(int)
    data_nan_indices = np.where(data < min_value, 1, 0)  # get NaN indices
    if _verbose:
        print(f'  --Missing data found: {np.sum(data_nan_indices)}')

    # Find indices of rows with NaNs
    nan_loc = {}
    for i in range(_row_pixels):
        if np.sum(data_nan_indices[i]) > 0:
            # Find the indices of columns with NaNs, save them in their corresponding row
            cols = np.where(data_nan_indices[i] == 1)
            nan_loc[i] = cols[0].tolist()
        
    # filled_data = data[:]

    fill_values = np.empty(data.shape)
    fill_values[:] = np.nan
    
    for row in nan_loc.keys():
        for col in nan_loc[row]:

            nan_value = data[row, col]  # value of the missing data
            
            # Get a window around the missing data pixel
            win_size = 1
            removed_success = False
            while not removed_success:
                # Window to slice around the missing value
                row_start = row-win_size
                row_end = row+win_size+1
                col_start = col-win_size
                col_end = col+win_size+1
                # Adjust row,col to use for slicing when point near the edges
                if _max_row is not None:
                    if row_start < 0:
                        row_start = 0
                    if row_end > _max_row:
                        row_end = _max_row
                if _max_col is not None:
                    if col_start < 0:
                        col_start = 0
                    if col_end > _max_col:
                        col_end = _max_col
                
                # Slice a window of values around missing value
                window_data = data[row_start:row_end, col_start:col_end]
                win_values = window_data.flatten().tolist()

                # Remove NaN values from the list
                all_values = win_values.copy()
                win_values = [i for i in all_values if i != nan_value]
                # If the list is not empty, then NaNs were removed successfully!
                if len(win_values) > 0:
                    removed_success = True
                    if _verbose:
                        print(f'  -- {_id}: Success with window size {win_size}. ({row},{col})')
                    break

                # If failure, increase window size and try again
                win_size += 1
            
            # Use mode as fill value (will return minimum value as default)
            fill_value = stats.mode(win_values, keepdims=False)[0]
            if fill_value == np.min(win_values):
                # If default (minimum) means not mode was found, return mean value instead
                fill_value = int(np.nanmean(win_values))
            if _verbose:
                print(f'  -- Fill value: {fill_value}')
            fill_values[row, col] = fill_value
    
    # Fill the missing values in their right position
    filled_data = np.where(data_nan_indices == 1, fill_values, data)

    return filled_data

def fix_annual_phenology(data: np.ndarray) -> np.ndarray:
        """Fix phenology values larger than 366 and returns the fixed dataset"""
        MAX_ITERS = 10
        iter = 0
        while np.max(data) > 366 or iter >= MAX_ITERS:
            data = np.where(data > 366, data-365, data)
            iter += 1
        return data

### ------ End of functions, start main code -----

if __name__ == '__main__':
    # --*-- TESTING CODE --*--

    # ############################### Create MODIS-like QA from Landsat ###################################
    # print('  Creating MODIS-like QA raking from LANDSAT')
    # directory = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/021046'

    # fn1 = 'LC08_L1TP_021046_20130415_20170310_01_T1_pixel_qa.tif'
    # fn_raster_qa = os.path.join(directory, fn1)
    # pixel_arr, pixel_nd, pixel_meta, pixel_gt = open_raster(fn_raster_qa)

    # fn2 = 'LC08_L1TP_021046_20130415_20170310_01_T1_sr_aerosol.tif'
    # fn_raster_aerosol = os.path.join(directory, fn2)
    # aerosol_arr, aerosol_nd, aerosol_meta, aerosol_gt = open_raster(fn_raster_aerosol)

    # print('  --Creating ranks...')
    # rank_qa = create_qa(pixel_arr, aerosol_arr)
    # print(f'  --Ranks: {np.unique(rank_qa)}')

    # # Create a personalized colormap for the QA Rank
    # col_dict={0:"green", 1:"blue", 2:"yellow", 3:"grey"}
    # lbls = np.array(["0", "1", "2", "3"])
    # save_cmap_array(rank_qa, fn1[:-13], col_dict, lbls, title='QA Rank', suffix='_rank')

    # fn = '/vipdata/2023/land_cover_analysis/data/qgis_cmap_landcover_CBR_viri.clr'
    # cmap = read_clr(fn)
    # print(f'  --Colors: {cmap.colors}')

    print("  Loading rsmodule.")
    sys.exit(0)

