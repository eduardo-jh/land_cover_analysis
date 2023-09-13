#!/usr/bin/env python
# coding: utf-8

""" Classes for land cover classification

Provides classes for every step of the land cover classification
based on Random Forests and a tiling system either training a single
classifier per tile or a classifier for the entire mosaic, then
predicts the land cover classes tile by tile.

It is possibile of save and load trained Random Forests.

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2023-07-24 13:52:18.860417295 -0700

Changelog:
  2023-09-10: main issue: currently predictions look like speckled images.

"""
import os
import gc
import csv
import random
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.colors as clrs
import matplotlib.ticker as tckr
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Dict
from scipy import stats
from datetime import datetime
from osgeo import gdal
from osgeo import osr
from pyhdf.SD import SD, SDC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.style.use('ggplot')  # R-like plots


class LandCoverRaster():
    """ Reads land cover from a raster file (GeoTIFF) """

    fmt = '%Y_%m_%d-%H_%M_%S'

    def __init__(self, fn_landcover, output_dir):

        assert os.path.isfile(fn_landcover), f"File not found: {fn_landcover}"
        assert os.path.isdir(output_dir), f"Directory not found: {output_dir}"

        self.start = datetime.now()

        self.output_dir = output_dir

        self.fn_landcover = fn_landcover
        self.dataset = None
        self.geotransform = None
        self.nodata = None
        self.spatial_reference = None
        self.nrows = None
        self.ncols = None
        self.shape = None
        self.dtype = None
        self.extent = None
        
        self.ancillary_dir = None
        self.ancillary_dict = None
        self.ancillary_suffix = None

        # For sampling
        self.train_percent = None
        self.window_size = None
        self.max_trials = None
        self.classes = None
        self.freqs = None
        self.percentages = None
        self.landcover_frequencies = None
        self.fig_frequencies = None
        self.sampling_suffix = None
        self.fn_training_mask = None
        self.fn_training_labels = None
        self.fn_sample_sizes = None

        self.basename, self.file_extension = os.path.splitext(fn_landcover)

        self.read_land_cover()

        print(f"\n*** Initializing land cover raster... done. ***\n")


    def __str__(self):
        text = "\nLand cover raster:\n"
        text += f"{'Raster file':<25} {self.fn_landcover}\n"
        text += f"{'Base file':<25} {self.basename}\n"
        text += f"{'File extension':<25} {self.file_extension}\n"
        text += f"{'Output directory':<25} {self.output_dir}\n"
        text += f"{'Geotransform':<25} {self.geotransform}\n"
        text += f"{'Extent':<25} {self.extent}\n"
        text += f"{'NoData':<25} {self.nodata}\n"
        text += f"{'Spatial reference (WKT)':<25} {self.spatial_reference}\n"
        text += f"{'Spatial reference (PROJ4)':<25} {self.spatial_reference.ExportToProj4()}\n"
        text += f"{'Rows':<25} {self.nrows}\n"
        text += f"{'Columns':<25} {self.ncols}\n"
        text += f"{'Shape':<25} {self.shape}\n"
        text += f"{'Data type':<25} {self.dtype}\n"
        text += f"{'Ancillary':<25} {self.ancillary_dict}\n"
        text += f"{'Ancillary suffix':<25} {self.ancillary_suffix}\n"
        text += f"{'Ancillary dir':<25} {self.ancillary_dir}\n"
        text += f"{'Train percent':<25} {self.train_percent}\n"
        text += f"{'Window size':<25} {self.window_size}\n"
        text += f"{'Max trials':<25} {self.max_trials}\n"
        text += f"{'Figure frequencies':<25} {self.fig_frequencies}\n"
        text += f"{'File sample sizes':<25} {self.fn_sample_sizes}\n"
        text += f"{'Land cover frequencies':<25} {self.landcover_frequencies}\n"
        text += f"{'Classes':<25} {self.classes}\n"
        text += f"{'Training mask file':<25} {self.fn_training_mask}\n"
        text += f"{'Training labels file':<25} {self.fn_training_labels}\n"

        
        return text


    def read_land_cover(self):
        """ Populates the raster data and parameters"""
        print(f"Reading raster: {self.fn_landcover}")
        self.dataset, self.nodata, self.geotransform, self.spatial_reference = self.open_raster(self.fn_landcover)
        self.nrows = self.dataset.shape[0]
        self.ncols = self.dataset.shape[1]
        self.shape = (self.nrows, self.ncols)
        self.nodata = int(self.nodata)
        self.dtype = self.dataset.dtype
        self.dataset = self.dataset.filled(self.nodata)  # replace masked constant "--" with zeros (NoData)
        
        self.get_extent()


    def open_raster(self, filename: str) -> Tuple:
        """ Open a GeoTIFF raster and return a numpy array

        :param str filename: the file name of the GeoTIFF raster to open
        :return raster: a masked array with NoData values masked out
        """

        dataset = gdal.OpenEx(filename)

        # metadata = dataset.GetMetadata()
        geotransform = dataset.GetGeoTransform()
        nodata = dataset.GetRasterBand(1).GetNoDataValue()
        raster_array = dataset.ReadAsArray()
        # projection = dataset.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=dataset.GetProjection())
        # epsg = proj.GetAttrValue('AUTHORITY',1)
        # print(epsg)
        
        # Mask 'NoData' values
        raster = np.ma.masked_values(raster_array, nodata)

        # Clean
        del(dataset)
        del(raster_array)
        gc.collect()

        return raster, nodata, geotransform, spatial_ref
    

    def create_raster(self, filename: str, data: np.ndarray, spatial_ref: str, geotransform: list, **kwargs) -> None:
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


    def get_extent(self):
        """ Creates extent from geotransform """
        ulx, xres, _, uly, _, yres = self.geotransform
        self.extent = [ulx, ulx + xres*self.ncols, uly, uly + yres*self.nrows]


    def incorporate_ancillary(self, **kwargs):
        """ Incorporates a list of ancillary rasters to its corresponding land cover class
        Ancillary data has to be 1's for data and 0's for NoData
        """
        self.ancillary_dir = kwargs.get("ancillary_dir", "")
        self.ancillary_dict = kwargs.get("ancillary_dict", None)
        self.ancillary_suffix = kwargs.get("suffix", "_ancillary.tif")

        for key in self.ancillary_dict.keys():
            for ancillary_file in self.ancillary_dict[key]:
                fn = os.path.join(self.ancillary_dir, ancillary_file)
                print(f"Incorporating ancillary raster: {fn}")
                assert os.path.isfile(fn), f"File not found: {fn}"
                
                # Read ancillary data
                ancillary, anc_nd, _, _ = self.open_raster(fn)
                
                # assert anc_spatref == self.spatial_reference, f"Spatial reference doesn't match: {anc_spatref} and {self.spatial_reference}"
                # assert anc_nd == self.nodata, f"NoData value doesn't match: {anc_nd} and {self.nodata}"
                assert ancillary.shape == self.dataset.shape, f"Shape doesn't match: {ancillary.shape} and {self.dataset.shape}"
                self.dataset = np.where(ancillary > 0, key, self.dataset)
        
        # Save the land cover with the integrated ancillary data
        self.fn_landcover = self.basename + self.ancillary_suffix
        # print(type(self.spatial_reference), self.spatial_reference)
        self.create_raster(self.fn_landcover, self.dataset, self.spatial_reference, self.geotransform)
        print(f"Land cover file is now: {self.fn_landcover}")
        self.read_land_cover()  # update land cover


    def land_cover_freq(self, **kwargs):
        """ Generates a single dictionary of land cover classes/groups and its pixel frequency """
        _verbose = kwargs.get('verbose', False)

        # First get the land cover keys in the array, then get their corresponding description
        keys, freqs = np.unique(self.dataset, return_counts=True)

        if _verbose:
            print(f'  --{keys}')
            print(f'  --{len(keys)} unique land cover classes/groups in ROI.')

        self.landcover_frequencies = {} 
        for key, freq in zip(keys, freqs):

            if type(key) is np.ma.core.MaskedConstant or key == self.nodata:
                if _verbose:
                    print(f'  --Skip the MaskedConstant object: {key}')
                continue
            self.landcover_frequencies[key] = freq


    def plot_land_cover_hbar(self, **kwargs):
        """ Create a horizontal bar plot to show the land cover """
        _title = kwargs.get('title', '')
        _xlabel = kwargs.get('xlabel', '')
        _ylabel = kwargs.get('ylabel', '')
        _xlims = kwargs.get('xlims', (0,100))

        plt.figure(figsize=(8, 12), constrained_layout=True)
        pl = plt.barh(self.classes, self.percentages)
        for bar in pl:
            value = bar.get_width()
            text = round(value, 4)
            if value > 0.01:
                text = round(value, 2)
            plt.annotate(text, xy=(value+0.1, bar.get_y()+0.25))
        
        if _title != '':
            plt.title(_title)
        if _xlabel != '':
            plt.xlabel(_xlabel)
        if _ylabel != '':
            plt.ylabel(_ylabel)
        if _xlims is not None:
            plt.xlim(_xlims)
        plt.savefig(self.fig_frequencies, bbox_inches='tight', dpi=150)
        plt.close()


    def configure_sampling_files(self, **kwargs):
        
        print("\n*** Configure sampling files ***\n")

        self.sampling_suffix = kwargs.get("sampling_suffix", "sampling")
        self.fn_training_mask = kwargs.get("training_mask", "training_mask.tif")
        self.fn_training_labels = kwargs.get("training_labels", "training_labels.tif")
        self.fn_sample_sizes = kwargs.get("sample_sizes", "dataset_sample_sizes.csv")
        self.fig_frequencies = kwargs.get("fig_frequencies", "class_fequencies.png")

        sampling_dir = os.path.join(self.output_dir, self.sampling_suffix)
        if not os.path.exists(sampling_dir):
                print(f"\nCreating new path: {sampling_dir}")
                os.makedirs(sampling_dir)
        
        self.fn_training_mask = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_mask)
        self.fn_training_labels = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_labels)
        self.fig_frequencies = os.path.join(self.output_dir, self.sampling_suffix, self.fig_frequencies)
        self.fn_sample_sizes = os.path.join(self.output_dir, self.sampling_suffix, self.fn_sample_sizes)


    def configure_sampling_files_timed(self, **kwargs):
        
        print("\n*** Configure sampling files based on time ***\n")

        self.sampling_suffix = kwargs.get("sampling_suffix", "sampling")
        self.fn_training_mask = kwargs.get("training_mask", "training_mask")
        self.fn_training_labels = kwargs.get("training_labels", "training_labels")
        self.fn_sample_sizes = kwargs.get("sample_sizes", "dataset_sample_sizes")
        self.fig_frequencies = kwargs.get("fig_frequencies", "class_fequencies")

        sampling_dir = os.path.join(self.output_dir, self.sampling_suffix)
        if not os.path.exists(sampling_dir):
                print(f"\nCreating new path: {sampling_dir}")
                os.makedirs(sampling_dir)
        
        self.fn_training_mask = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_mask + f"_{datetime.strftime(self.start, self.fmt)}" + ".tif")
        self.fn_training_labels = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_labels + f"_{datetime.strftime(self.start, self.fmt)}" + ".tif")
        self.fig_frequencies = os.path.join(self.output_dir, self.sampling_suffix, self.fig_frequencies + f"_{datetime.strftime(self.start, self.fmt)}" + ".png")
        self.fn_sample_sizes = os.path.join(self.output_dir, self.sampling_suffix, self.fn_sample_sizes + f"_{datetime.strftime(self.start, self.fmt)}" + ".csv")


    def sample(self, **kwargs):
        """ Stratified random sampling

        :param float train_percent: default training-testing proportion is 80-20%
        :param int win_size: default is sampling a window of 7x7 pixels
        :param int max_trials: max of attempts to fill the sample size
        """
        
        self.train_percent = kwargs.get("train_percent", 0.2)
        self.window_size = kwargs.get("window_size", 7)
        self.max_trials = int(kwargs.get("max_trials", 2e5))

        self.configure_sampling_files()

        # Create a list of land cover keys and its area covered percentage
        self.land_cover_freq(verbose=False)
        print(f'  --Land cover freqencies: {self.landcover_frequencies}')
        
        self.classes = list(self.landcover_frequencies.keys())
        self.freqs = [self.landcover_frequencies[x] for x in self.classes]  # pixel count
        self.percentages = (self.freqs/sum(self.freqs))*100

        # Plot land cover percentage horizontal bar
        print('  --Plotting land cover percentages...')
        self.plot_land_cover_hbar(
            title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
            xlabel='Percentage (based on pixel count)',
            ylabel='Land Cover (Grouped)',  # remove if not grouped
            xlims=(0,100))

        #### Sample size == testing dataset
        # Use a dataframe to calculate sample size
        df = pd.DataFrame({'Key': self.classes, 'PixelCount': self.freqs, 'Percent': self.percentages})
        df['TrainPixels'] = (df['PixelCount']*self.train_percent).astype(int)
        # print(df['TrainPixels'])

        # Now calculate percentages
        df['TrainPercent'] = (df['TrainPixels'] / df['PixelCount'])*100
        print(df)

        print(f"  --Total pixels={self.nrows*self.ncols}, Values={sum(df['PixelCount'])}, NoData/Missing={self.nrows*self.ncols - sum(df['PixelCount'])}")

        sample = {}  # to save the sample

        # Create a mask of the sampled regions
        self.training_mask = np.zeros(self.dataset.shape, dtype=self.dataset.dtype)

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
            col_sample = random.randrange(0 + self.window_size//2, self.ncols - self.window_size//2, self.window_size)
            row_sample = random.randrange(0 + self.window_size//2, self.nrows - self.window_size//2, self.window_size)

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
            if win_col_end > self.ncols:
                # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
                win_col_end = self.ncols
            if win_row_ini < 0:
                # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
                win_row_ini = 0
            if  win_row_end > self.nrows:
                # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
                win_row_end = self.nrows

            # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
            window_sample = self.dataset[win_row_ini:win_row_end,win_col_ini:win_col_end]
            # print(window_sample)
            
            # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
            sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
            classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

            # 6) Iterate over each class sample and add its respective pixel count to the sample
            for sample_class, class_count in zip(sample_keys, sample_freq):
                if sample_class == self.nodata:
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
            sampled_window = np.zeros(window_sample.shape, dtype=self.dataset.dtype)
            
            # Filter out classes with already complete samples
            if len(classes_to_remove) > 0:
                for single_class in classes_to_remove:
                    # Put a 1 on a complete class
                    filter_out = np.where(window_sample == single_class, 1, 0)
                    sampled_window += filter_out.astype(self.dataset.dtype)
                
                # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
                sampled_window = np.where(sampled_window == 0, 1, 0)
            else:
                sampled_window = window_sample[:,:].astype(self.dataset.dtype)
            
            # Slice and insert sampled window
            # print(self.training_mask[win_row_ini:win_row_end,win_col_ini:win_col_end].dtype, sampled_window.dtype)
            self.training_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window.astype(self.dataset.dtype)

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

        # Create a raster with actual labels (land cover classes)
        self.training_labels = np.where(self.training_mask > 0, self.dataset, 0)
        print(f"Creating raster: {self.fn_training_labels}")
        self.create_raster(self.fn_training_labels, self.training_labels, self.spatial_reference, self.geotransform)

        # Create a raster with the sampled windows, this will be the sampling mask
        print(f"Creating raster: {self.fn_training_mask}")
        self.create_raster(self.fn_training_mask, self.training_mask, self.spatial_reference, self.geotransform)

        self.end = datetime.now()
        print(f"Sampling elapsed in: {self.end - self.start}.")



class FeaturesDataset():
    """ Reads spectral bands and phenology HDF4 files and creates HDF5 feature files """

    FEAT_OFFSET = 4  # not part of feature name
    PHENO_OFFSET = 5
    BAND_SPECTRAL = 0
    BAND_VI = 1
    BAND_PHENOLOGY_1 = 2
    BAND_PHENOLOGY_2 = 3
    

    def __init__(self, raster, bands_dir, pheno_dir, **kwargs):
        self.features_suffix = kwargs.get("features_dir", "features")
        self.labels_suffix = kwargs.get("labels_suffix", "data/inegi")
        self.fn_tiles = kwargs.get("file_tiles", None)
        # self.phenobased = kwargs.get("phenobased", "NDVI")
        self.phenobased = kwargs.get("phenobased", "")
        self.feat_list = kwargs.get("feat_list", None)

        assert os.path.isdir(bands_dir), f"Directory not found: {bands_dir}"
        assert os.path.isdir(pheno_dir), f"Directory not found: {pheno_dir}"
        # assert os.path.isfile(fn_landcover), f"File not found: {fn_landcover}"
        
        self.land_cover_raster = raster

        self.ds_nrows = None
        self.ds_ncols = None
        self.tiles = None
        self.tiles_extent = None
        self.tiles_slice = None
        self.training_mask = None
        self.no_data_arr = None

        if self.land_cover_raster.fn_training_mask:
            assert os.path.isfile(self.land_cover_raster.fn_training_mask), f"File not found: {self.land_cover_raster.fn_training_mask}"
            self._read_training_mask()
        if self.fn_tiles:
            print(f"File with tiles passed: {fn_tiles}")
            assert os.path.isfile(self.fn_tiles), f"File not found: {self.fn_tiles}"
            self._read_tiles()
            self._tile_extent_2_slice()

        self.bands_dir = bands_dir
        self.pheno_dir = pheno_dir

        self.feat_indices = None
        self.feat_names = None
        self.feat_type = None

        self.months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        self.variables = ['AVG', 'STDEV']
        self.phenology = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
        self.phenology2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

        self.fill = None
        self.standardize = None
        self.normalize = None

        self.fn_feature_labels = None

        self.feats_by_season = None
        self.feat_indices_season = None
        self.feat_names_season = None

        # self.read_params_from_features()

        # Create a dataset of labels for the entire mosaic
        self.fn_feature_labels = os.path.join(self.land_cover_raster.output_dir, self.features_suffix, f"land_cover_labels_{self.phenobased}.h5")


        print("Initializing directories to parse HDF... done!")


    def __str__(self):
        text = "\n"
        text += self.land_cover_raster.__str__()
        text += "\nFeatures dataset:\n"
        text += f"{'Spectral bands directory':<25} {self.bands_dir}\n"
        text += f"{'Phenology directory':<25} {self.pheno_dir}\n"
        text += f"{'Features suffix':<25} {self.features_suffix}\n"
        text += f"{'Labels suffix':<25} {self.labels_suffix}\n"
        text += f"{'File with tiles':<25} {self.fn_tiles}\n"
        text += f"{'Phenology based':<25} {self.phenobased}\n"
        text += f"{'List of features':<25} {self.feat_list}\n"
        # text += f"{'Training mask file':<25} {self.land_cover_raster.fn_training_mask}\n"
        text += f"{'Feature indices':<25} {self.feat_indices}\n"
        text += f"{'Feature names':<25} {self.feat_names}\n"
        text += f"{'Feature types':<25} {self.feat_type}\n"
        text += f"{'Months':<25} {self.months}\n"
        text += f"{'Variables':<25} {self.variables}\n"
        text += f"{'Phenology':<25} {self.phenology}\n"
        text += f"{'Phenology2':<25} {self.phenology2}\n"
        text += f"{'Fill':<25} {self.fill}\n"
        text += f"{'Standardize':<25} {self.standardize}\n"
        text += f"{'Normalize':<25} {self.normalize}\n"
        text += f"{'Dataset rows':<25} {self.ds_nrows}\n"
        text += f"{'Dataset columns':<25} {self.ds_ncols}\n"
        text += f"{'Tiles':<25} {self.tiles}\n"

        return text


    def read_params_from_features(self):
        """Initializes parameters in case user does not run functions to generate datasets"""

        print("\nConfigure parameters without generating datasets.")

        # self._generate_feature_list() # will generate monthly features only
        
        # Check number of features with a single tile (the first one)
        fn_dummy_tile = os.path.join(self.land_cover_raster.output_dir, self.features_suffix, self.phenobased, self.tiles[0], f"features_season_{self.tiles[0]}.h5")
        # First try to generate features by season
        if not os.path.isfile(fn_dummy_tile):
            # In case seasonal features not available, will generate monthly features
            fn_dummy_tile = os.path.join(self.land_cover_raster.output_dir, self.features_suffix, self.phenobased, self.tiles[0], f"features_{self.tiles[0]}.h5")
        if not os.path.isfile(fn_dummy_tile):
            print(f"*** Warning: parameters not configured! File not found: {fn_dummy_tile} ***")
            return
        
        # Read the features to initialize
        with h5py.File(fn_dummy_tile, 'r') as h5_features:
            dummy_features = [key for key in h5_features.keys()]
            for i, feature in enumerate(dummy_features):
                print(f" Feature {i}: {feature}")
            # Set rows and columns
            dummy = h5_features[dummy_features[0]][:]
            print(f"Dummy shape: {dummy.shape}")
            self.ds_nrows, self.ds_ncols = dummy.shape
            if self.feat_names is None:
                print("Generating features from existing dataset.")
                self.feat_names = dummy_features
                self.feat_indices = [x for x, _ in enumerate(self.feat_names)]
                self._get_feat_type()
            del dummy


    def get_output_dir(self):
        return self.land_cover_raster.output_dir
    
    
    def _read_from_hdf(self, filename: str, var: str, dtype: np.dtype) -> np.ndarray:
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

        data_arr = np.round(data_arr).astype(dtype)

        return data_arr
    

    def _get_band(self, feature_name: str) -> str:
        """Retrieves the band from a feature name and returns it as a string"""
        band = ''
        feat_nameparts = feature_name.split(' ')
        if len(feat_nameparts) == 2:
            band = feat_nameparts[1]
        elif len(feat_nameparts) == 3:
            band = feat_nameparts[1]
        elif len(feat_nameparts) == 4:
            band = feat_nameparts[2][1:-1]  # remove parenthesis
        return band
    

    def _fix_annual_phenology(self, data: np.ndarray) -> np.ndarray:
        """Fix phenology values larger than 366 and returns the fixed dataset"""
        MAX_ITERS = 10
        iter = 0
        while np.max(data) > 366 or iter >= MAX_ITERS:
            data = np.where(data > 366, data-365, data)
            iter += 1
        return data
    

    def _normalize_dataset(self, ds: np.ndarray) -> np.ndarray:
        """ Normalize a dataset with min-max feature scaling into range [0,1] """
        _min = np.nanmin(ds)
        _max = np.nanmax(ds)
        return (ds - _min) / (_max - _min)


    def _standardize_dataset(self, ds: np.ndarray) -> np.ndarray:
        """ Standarize a dataset into range [-1, 1] """
        avg = np.nanmean(ds)
        std = np.nanstd(ds)
        return (ds - avg) / std
    

    def _fill_with_mean(self, dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def _fill_with_mode(self, data: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def _fill_with_int_mean(self, dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def _fill_season(self, sos: np.ndarray, eos: np.ndarray, los: np.ndarray, min_value: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


    def _generate_feature_list(self):
        """Generates the list of features (names and indices) to be used"""

        self.feat_indices = []
        self.feat_names = []

        if type(self.feat_list) is list:
            # A list of features is received
            for i, feat in enumerate(self.feat_list):
                self.feat_indices.append(i)
                self.feat_names.append(feat)
        elif type(self.feat_list) is str:
            # Look for features from a text file
            assert os.path.isfile(self.feat_list), f"File not found: {self.feat_list}"
            print(f'Trying to get features from: {self.feat_list}')
            feature = 0
            content = ""
            with open(self.feat_list, 'r') as f:
                content = f.readlines()
            for i, line in enumerate(content):
                line = line.strip()
                if line != "":
                    self.feat_names.append(line)
                    self.feat_indices.append(feature)
                    print(f"{feature}: {line}")
                    feature += 1
        else:
            # No features provided, combine bands and variables (statistics) to generate features
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
            elif self.phenobased == "":
                bands = ['Blue', 'Evi', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
                band_num = ['B2', '','B3', 'B7', '', 'B5', 'B4', 'B6']

            # Generate feature names...
            print('Generating feature names from combination of variables.')
            feature = 0
            for j, band in enumerate(bands):
                for i, month in enumerate(self.months):
                    for var in self.variables:
                        # Create the name of the dataset in the HDF
                        feat_name = month + ' ' + band_num[j] + ' (' + band + ') ' + var
                        if band_num[j] == '':
                            feat_name = month + ' ' + band.upper() + ' ' + var
                        # print(f'  Feature: {feature} Variable: {feat_name}')
                        self.feat_names.append(feat_name)
                        self.feat_indices.append(feature)
                        feature += 1
            for param in self.phenology + self.phenology2:
                feat_name = 'PHEN ' + param
                # print(f'  Feature: {feature} Variable: {feat_name}')
                self.feat_names.append(feat_name)
                self.feat_indices.append(feature)
                feature += 1
        
        self._get_feat_type()
        

    def _get_feat_type(self):
        self.feat_type = []
        # Identify the type of data
        for feat in self.feat_names:
            band = self._get_band(feat)
            band_type = self.BAND_SPECTRAL
            if band in ["NDVI", "EVI", "EVI2"]:
                band_type = self.BAND_VI
            elif band in self.phenology:
                band_type = self.BAND_PHENOLOGY_1
            elif band in self.phenology2:
                band_type = self.BAND_PHENOLOGY_2
            self.feat_type.append(band_type)
            

    def _read_tiles(self):
        """ Put the extent of tiles into a dictionary 
        Extent format is a dictionary with N, S, W, and E boundaries
        """
        self.tiles_extent = {}
        self.tiles = []
        print(f"Read tiles extent from file:")
        with open(self.fn_tiles, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                print(row)
                row_dict = {}
                for item in row[1:]:
                    itemlst = item.split('=')
                    row_dict[itemlst[0].strip()] = int(float(itemlst[1]))
                self.tiles_extent[row[0]] = row_dict
                self.tiles.append(row[0])
        # print(self.tiles)


    def _read_training_mask(self):
        """ Opens raster sample mask, doesn't change other parameters """
        # WARNING! Assumes spatial reference is same as object's default!
        self.training_mask, _, _, _ = self.land_cover_raster.open_raster(self.land_cover_raster.fn_training_mask)
        self.training_mask = self.training_mask.astype(self.land_cover_raster.dtype)

    
    def _tile_extent_2_slice(self):
        """ Transforms extent from coordinates to columns and rows to slice the dataset """

        self.tiles_slice = {}

        # Extent will be N-S, and W-E boundaries
        raster_extent = {}
        raster_extent['W'], xres, _, raster_extent['N'], _, yres = [int(x) for x in self.land_cover_raster.geotransform]
        raster_extent['E'] = raster_extent['W'] + self.land_cover_raster.ncols*xres
        raster_extent['S'] = raster_extent['N'] + self.land_cover_raster.nrows*yres
        print(raster_extent)

        print(f"Tile extent into slice coordinates.")
        for tile in self.tiles:
            # Calculate slice coodinates to extract the tile
            tile_ext = self.tiles_extent[tile]

            # Get row for Nort and South and column for West and East
            tile_extent_slice = {}
            tile_extent_slice['N'] = (tile_ext['N'] - raster_extent['N'])//yres  # north row = top
            tile_extent_slice['S'] = (tile_ext['S'] - raster_extent['N'])//yres  # south row = bottom
            tile_extent_slice['W'] = (tile_ext['W'] - raster_extent['W'])//xres  # west column = left
            tile_extent_slice['E'] = (tile_ext['E'] - raster_extent['W'])//xres  # east colum = right
            print(tile_extent_slice)

            # Save the colums and rows to slice the tile dataset
            self.tiles_slice[tile] = tile_extent_slice
    

    def _generate_season_feature_list(self):
        """ List of features grouped by season """

        seasons = {'SPR': ['APR', 'MAY', 'JUN'],
                'SUM': ['JUL', 'AUG', 'SEP'],
                'FAL': ['OCT', 'NOV', 'DEC'],
                'WIN': ['JAN', 'FEB', 'MAR']}
        
        # Get the unique bands and variables from feature names
        bands = []
        vars = []
        for feature in self.feat_names:
            band = self._get_band(feature)
            var = feature.split(" ")[-1]
            if band not in bands:
                bands.append(band)
            if var not in vars:
                vars.append(var)

        # Group feature names by season -> band -> variable -> month
        self.feats_by_season = {}
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
                                if self.feats_by_season.get(season_key) is None:
                                    self.feats_by_season[season_key] = [feat_name]
                                else:
                                    self.feats_by_season[season_key].append(feat_name)
                                # print(f"  -- {season} {band:>5} {var:>5}: {feat_name}")
        self.feat_indices_season = []
        self.feat_names_season = []
        feat_num = 0

        # Calculate averages of features grouped by season
        for key in list(self.feats_by_season.keys()):
            print(f"  *{key:>15}:")
            for i, feat_name in enumerate(self.feats_by_season[key]):
                print(f"   -Adding {feat_num}: {feat_name}")
               

            self.feat_indices_season.append(feat_num)
            self.feat_names_season.append(key)

            feat_num += 1

        # Add PHEN features directly, no aggregation by season
        for feat_name in self.feat_names:
            if feat_name[:4] == 'PHEN':
                print(f"   -Adding {feat_num}: {feat_name}")

                self.feat_indices_season.append(feat_num)
                self.feat_names_season.append(feat_name)

                feat_num += 1


    def create_features_dataset(self, **kwargs):
        """ Decides between creating a single dataset or a tiled mosaic dataset
            depnding whether or not a list of tiles was provided as self.tiles
        """
        _by_season = kwargs.get("by_season", False)
        _save_labels_raster = kwargs.get('save_labels_raster', False)
        _save_features = kwargs.get('save_features', False)

        if self.tiles is None:
            # No tiles provided, creating a single dataset
            #TODO: actually implement this part!
            print("Creating single dataset with same dimensions as raster.")
            self._create_features_single()
        else:
            print("Creating a dataset per tile.")
            self._create_features_mosaic(by_season=_by_season,
                                       save_labels_raster=_save_labels_raster,
                                       save_features=_save_features)


    def _create_features_single(self):
        """Create the fearut"""
        pass


    def _create_features_mosaic(self, **kwargs) -> None:
        save_labels_raster = kwargs.get('save_labels_raster', False)
        save_features = kwargs.get('save_features', False)
        by_season = kwargs.get('by_season', False)

        if by_season:
            # This option in mandatory in this case
            save_features = True
            
        # First, create the list of features
        self._generate_feature_list()

        # Genereate the list of features by season if required
        if by_season and self.feat_names_season is None:
            self._generate_season_feature_list()
        
        assert self.tiles is not None, "List of tiles is empty (None)."

        # In case sampling was executed in a previous run
        if self.training_mask is None:
            self._read_training_mask()
        
        for tile in self.tiles:
            print(f"\nProcessing tile: {tile}")
            feat_path = os.path.join(self.land_cover_raster.output_dir, self.features_suffix, self.phenobased, tile)
            if not os.path.exists(feat_path) and save_features:
                print(f"\nCreating new path: {feat_path}")
                os.makedirs(feat_path)
            
            # Generate features
            tile_features = self._generate_features_array(tile, fill=False)
            
            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            
            if save_features:
                print(f"Saving {tile} features...")
                # Create (large) HDF5 files to hold all features
                h5_features = h5py.File(fn_tile_features, 'w')
                for n, feature in zip(self.feat_indices, self.feat_names):
                    h5_features.create_dataset(feature, (self.ds_rows, self.ds_cols), data=tile_features[:,:,n])

            if by_season:
                # Aggregate features by season
                # WARNING! Requires HDF5 files to be created first!
                fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")
                self._group_features_by_season(fn_tile_features, fn_tile_feat_season)

    
    def _generate_features_array(self, tile: str, **kwargs) -> np.ndarray:
        """Reads the STATS and PHENOLOGY HDF4 files from their corresponding tile directory
           and creates an array with all the features.
           
        :param str tile: the name of the tile to generate the dataset
        :return: numpy array with the features
        """
    # def generate_features_array(self, tile, ds_shape, **kwargs):
        dtype = kwargs.get("dtype", np.int16)
        fill = kwargs.get("fill", False)
        normalize = kwargs.get("normalize", False)
        standardize = kwargs.get("standardize", False)
        nodata_filter = kwargs.get("nodata_mask", None)

        self.fill = fill
        self.standardize = standardize
        self.normalize = normalize

        # If no tile dimensions provided, read from the actual HDF4 file
        # WARNING! Hardcoded names are required in this case!
        fn_dummy = os.path.join(self.bands_dir, self.tiles[0], "MONTHLY.BLUE.APR.hdf")
        dummy_ds = self._read_from_hdf(fn_dummy, 'B2 (Blue) AVG', np.int16)
        self.ds_nrows, self.ds_ncols = dummy_ds.shape

        # Array to hold all features
        print(self.ds_nrows, self.ds_nrows, self.feat_indices)
        features = np.zeros((self.ds_nrows, self.ds_nrows, len(self.feat_indices)), dtype=dtype)

        for feat_index, feat_name, feat_type in zip(self.feat_indices, self.feat_names, self.feat_type):
            feat_name_parts = feat_name.split(' ')

            if feat_type == self.BAND_PHENOLOGY_1:
                # Phenology 1
                feat = feat_name[self.PHENO_OFFSET:]  # actual feature name
                
                fn = os.path.join(self.pheno_dir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S1.hdf")
                if self.phenobased == '':
                    fn = os.path.join(self.pheno_dir, 'NDVI', tile, f"LANDSAT08.PHEN.NDVI_S1.hdf")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                print(f"--{feat_index:>3}: {feat} --> {fn}")

                # No need to fill missing values, just read the values
                feat_arr = self._read_from_hdf(fn, feat, np.int16)  # Use HDF4 method
                # print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.land_cover_raster.nodata)
                
                # Fix values larget than 366
                if feat == 'SOS' or feat == 'EOS' or feat == 'LOS':
                    print(f' --Fixing {feat}.')
                    feat_fixed = self._fix_annual_phenology(feat_arr)
                    feat_arr = feat_fixed[:]

                # Fill missing data
                if self.fill:
                    if feat == 'SOS':
                        print(f'  --Filling {feat}')
                        minimum = 0
                        sos = self._read_from_hdf(fn, 'SOS', np.int16)
                        eos = self._read_from_hdf(fn, 'EOS', np.int16)
                        los = self._read_from_hdf(fn, 'LOS', np.int16)
                        
                        sos_fixed = self._fix_annual_phenology(sos)
                        eos_fixed = self._fix_annual_phenology(eos)
                        los_fixed = self._fix_annual_phenology(los)
                        
                        # # Fix SOS values larger than 365
                        # sos_fixed = np.where(sos > 366, sos-365, sos)

                        # # Fix SOS values larger than 365, needs to be done two times
                        # eos_fixed = np.where(eos > 366, eos-365, eos)
                        # # print(np.min(eos_fixed), np.max(eos_fixed))
                        # if np.max(eos_fixed) > 366:
                        #     eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
                        #     print(f'  --Adjusting EOS again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')

                        filled_sos, filled_eos, filled_los =  self._fill_season(sos_fixed, eos_fixed, los_fixed, minimum,
                                                                        row_pixels=self.ds_nrows,
                                                                        max_row=self.ds_nrows,
                                                                        max_col=self.ds_ncols,
                                                                        id=feat,
                                                                        verbose=False)

                        feat_arr = filled_sos[:]
                    elif feat == 'EOS':
                        print(f'  --Filling {feat}')
                        feat_arr = filled_eos[:]
                    elif feat == 'LOS':
                        print(f'  --Filling {feat}')
                        feat_arr = filled_los[:]
                    elif feat == 'DOP' or feat == 'NOS':
                        # Day-of-peak and Number-of-seasons, use mode
                        print(f'  --Filling {feat}')
                        feat_arr = self._fill_with_mode(feat_arr, 0, row_pixels=self.tile_rows, max_row=self.tile_rows, max_col=self.tile_cols, verbose=False)
                    elif feat == 'GDR' or feat == 'GUR' or feat == 'MAX':
                        # GDR, GUR and MAX should be positive integers!
                        print(f'  --Filling {feat}')
                        feat_arr = self._fill_with_int_mean(feat_arr, 0, var=feat, verbose=False)
                    else:
                        # Other parameters? Not possible
                        print(f'  --Filling {feat}')
                        ds = self._read_from_hdf(fn, feat, np.int16)
                        feat_arr = self._fill_with_int_mean(ds, 0, var=feat, verbose=False)

                # Normalize or standardize
                assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                if self.normalize and not self.standardize:
                    feat_arr = self._normalize_dataset(feat_arr)
                elif not self.normalize and self.standardize:
                    feat_arr = self._standardize_dataset(feat_arr)
            
            elif feat_type == self.BAND_PHENOLOGY_2:
                # Phenology 2
                feat = feat_name[self.PHENO_OFFSET:]

                fn = os.path.join(self.pheno_dir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S2.hdf")
                if self.phenobased == '':
                    fn = os.path.join(self.pheno_dir, 'NDVI', tile, f"LANDSAT08.PHEN.NDVI_S2.hdf")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                print(f"--{feat_index:>3}: {feat} --> {fn}")

                # No need to fill missing values, just read the values
                feat_arr = self._read_from_hdf(fn, feat, np.int16)  # Use HDF4 method
                # print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.land_cover_raster.nodata)

                # Fix values larget than 366
                if feat == 'SOS' or feat == 'EOS' or feat == 'LOS':
                    print(f' --Fixing {feat}.')
                    feat_fixed = self._fix_annual_phenology(feat_arr)
                    feat_arr = feat_fixed[:]

                # Extract data and filter by training mask
                if self.fill:
                    # IMPORTANT: Only a few pixels have a second season, thus dataset could
                    # have a huge amount of NaNs, filling will be restricted to replace a
                    # The missing values to NO_DATA
                    print(f'  --Filling {feat}')
                    feat_arr = self._read_from_hdf(fn, feat, np.int16)
                    feat_arr = np.where(feat_arr <= 0, self.land_cover_raster.nodata, feat_arr)
                
                # Normalize or standardize
                assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                if self.normalize and not self.standardize:
                    feat_arr = self._normalize_dataset(feat_arr)
                elif not self.normalize and self.standardize:
                    feat_arr = self._standardize_dataset(feat_arr)
            elif feat_type == self.BAND_SPECTRAL or feat_type == self.BAND_VI:
                # VI or SPECTRAL BAND feature
                feat = feat_name[self.FEAT_OFFSET:]

                if len(feat_name_parts) == 3:
                    month = feat_name_parts[0]
                    band = feat_name_parts[1]
                    fn = os.path.join(self.bands_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')
                elif len(feat_name_parts) == 4:
                    month = feat_name_parts[0]
                    band = feat_name_parts[2][1:-1]  # remove parenthesis
                    fn = os.path.join(self.bands_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')

                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                print(f"--{feat_index:>3}: {feat} --> {fn}")

                # Extract data and filter
                feat_arr = self._read_from_hdf(fn, feat, np.int16)  # Use HDF4 method
                # print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.land_cover_raster.nodata)

                ### Fill missing data
                if self.fill:
                    minimum = 0  # set minimum for spectral bands
                    if feat_type == self.BAND_VI:
                        minimum = -10000  # minimum for VIs
                    feat_arr = self._fill_with_mean(feat_arr, minimum, var=band.upper(), verbose=False)

                # Normalize or standardize
                if self.normalize:
                    feat_arr = normalize(feat_arr)

            features[:,:,feat_index] = feat_arr
        return features
    

    def _group_features_by_season(self, fn_features, fn_features_season):
        """ Aggregates an HDF5 file with monthly features into another HDF5 file but with seasonal features 
        :param str fn_features: the input file name of the HDF5 file with monthly features
        :param str fn_features_season: the output file name of the HDF5 file with seasonal features
        """

        assert os.path.isfile(fn_features), f"ERROR: File not found: {fn_features}"

        h5_features = h5py.File(fn_features, 'r')
        h5_features_season = h5py.File(fn_features_season, 'w')

        if self.feat_names_season is None:
            self._generate_season_feature_list()

        feat_num = 0

        # Calculate averages of features grouped by season
        for key in list(self.feats_by_season.keys()):
            print(f"  *{key:>15}:")
            for i, feat_name in enumerate(self.feats_by_season[key]):
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
            # feat_arr /= len(self.feats_by_season[key])
            feat_arr = np.round(np.round(feat_arr).astype(np.int16) / np.int16(len(self.feats_by_season[key]))).astype(np.int16)

            h5_features_season.create_dataset(key, feat_arr.shape, data=feat_arr)

            # self.feat_indices_season.append(feat_num)
            # self.feat_names_season.append(key)

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

                # self.feat_indices_season.append(feat_num)
                # self.feat_names_season.append(feat_name)

                feat_num += 1
        
        print(f"File: {fn_features_season} created successfully.")


    def create_labels_dataset(self):
        """CAUTION: Does not work as expected"""
        # Read the raster
        print("\n *** Creating labels dataset ***")
        print(f"Reading labels raster: {self.land_cover_raster.fn_landcover}")
        print(f"Reading training mask: {self.land_cover_raster.fn_training_mask}")

        self.land_cover_raster.read_land_cover()
        self._read_training_mask()

        self.no_data_arr = np.where(self.land_cover_raster.dataset > self.land_cover_raster.nodata, 1, self.land_cover_raster.nodata)  # 1=data, 0=NoData
        # TODO: try removing this to avoid converting to unsigned int 8-bit
        # self.no_data_arr = self.no_data_arr.astype(self.land_cover_raster.dtype)
        # Keep train mask values only in pixels with data, remove NoData
        print(f"  Train mask shape: {self.training_mask.shape}")
        self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.land_cover_raster.nodata)

        # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
        print(f"  no_data_arr:   shape={self.no_data_arr.shape} type={self.no_data_arr.dtype} values={np.unique(self.no_data_arr, return_counts=True)}")
        print(f"  training_mask: shape={self.training_mask.shape} type={self.training_mask.dtype} values={np.unique(self.training_mask, return_counts=True)}")
        # print(f'  --Training pixels: {(self.training_mask == 1).sum()}')
        # print(f'  --No Data pixels: {(self.no_data_arr == 0).sum()}')
        # print(f'  --Testing pixels: {(self.training_mask == 0).sum() - (self.no_data_arr == 0).sum()}')

        print(f"Creating mosaic labels: {self.fn_feature_labels}")
        h5_all_labels = h5py.File(self.fn_feature_labels, 'w')
        # h5_all_labels.create_dataset('all', self.land_cover_raster.dataset.shape, data=self.land_cover_raster.dataset)
        h5_all_labels.create_dataset('land_cover', self.land_cover_raster.dataset.shape, data=self.land_cover_raster.dataset)
        h5_all_labels.create_dataset('training_mask', self.land_cover_raster.dataset.shape, data=self.training_mask)
        h5_all_labels.create_dataset('no_data_mask', self.land_cover_raster.dataset.shape, data=self.no_data_arr)
        # TODO: try removing this to avoid converting to unsigned int 8-bit
        # h5_all_labels.create_dataset('training_mask', self.land_cover_raster.dataset.shape, data=self.training_mask.astype(self.land_cover_raster.dtype))
        # h5_all_labels.create_dataset('no_data_mask', self.land_cover_raster.dataset.shape, data=self.no_data_arr.astype(self.land_cover_raster.dtype))

        # # h5_all_labels.create_dataset('all', self.land_cover_raster.dataset.shape, data=self.land_cover_raster.dataset, dtype=self.land_cover_raster.dtype)
        # # h5_all_labels.create_dataset('training_mask', self.land_cover_raster.dataset.shape, data=self.training_mask, dtype=self.land_cover_raster.dtype)
        # # h5_all_labels.create_dataset('no_data_mask', self.land_cover_raster.dataset.shape, data=self.no_data_arr, dtype=self.land_cover_raster.dtype)
        print("Labels dataset created successfully!\n")


    def create_tile_dataset(self, **kwargs) -> None:
        """Creates both features and labels dataset for a single tile, replaces the need to use
        'create_features_dataset' and 'create_labels_dataset' functions together"""
        save_labels_raster = kwargs.get('save_labels_raster', False)
        save_features = kwargs.get('save_features', False)
        by_season = kwargs.get('by_season', False)
        self.ds_nrows = kwargs.get('tile_rows', 5000)
        self.ds_ncols = kwargs.get('tile_cols', 5000)
        # labels_suffix = kwargs.get('labels_suffix', '') # labels_suffix='data/inegi'
        # feat_suffix = kwargs.get('feat_suffix', '')  # feat_suffix='features'

        if by_season:
            # This option in mandatory in this case
            save_features = True
            
        # # First, get tiles extent
        # self.read_tiles()
        
        # First, create the list of features
        self._generate_feature_list()

        # Genereate the list of features by season if required
        if by_season and self.feat_names_season is None:
            self._generate_season_feature_list()

        assert self.tiles is not None, "List of tiles is empty (None)."

        # In case sampling was executed in a previous run
        if self.training_mask is None:
            self._read_training_mask()
        # if self.training_labels is None:
        #     self.read_training_labels()

        # self.no_data_arr = np.where(self.dataset > self.nodata, 1, self.nodata)  # 1=data, 0=NoData
        self.no_data_arr = np.where(self.land_cover_raster.dataset > self.land_cover_raster.nodata, 1, self.land_cover_raster.nodata)  # 1=data, 0=NoData

        # Keep train mask values only in pixels with data, remove NoData
        print(f" Train mask shape: {self.training_mask.shape}")
        # self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.nodata)
        self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.land_cover_raster.nodata)


        # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
        print(f"  --no_data_arr={self.no_data_arr.dtype}, training_mask={self.training_mask.dtype} ")
        print(f'  --Training pixels: {(self.training_mask == 1).sum()}')
        print(f'  --No Data pixels: {(self.no_data_arr == 0).sum()}')
        print(f'  --Testing pixels: {(self.training_mask == 0).sum() - (self.no_data_arr == 0).sum()}')

        for tile in self.tiles:
            print(f"\nProcessing tile: {tile}")
            
            # Create new directories
            # labels_path = os.path.join(self.output_dir, labels_suffix, tile)
            # feat_path = os.path.join(self.output_dir, feat_suffix, self.phenobased, tile)
            labels_path = os.path.join(self.land_cover_raster.output_dir, self.labels_suffix, self.phenobased, tile)
            feat_path = os.path.join(self.land_cover_raster.output_dir, self.features_suffix, self.phenobased, tile)
            if not os.path.exists(labels_path) and save_labels_raster:
                print(f"\nCreating labels path: {labels_path}")
                os.makedirs(labels_path)
            if not os.path.exists(feat_path) and save_features:
                print(f"\nCreating features path: {feat_path}")
                os.makedirs(feat_path)

            # Create new file names
            fn_base = os.path.basename(self.land_cover_raster.fn_landcover)
            fn_tile = os.path.join(labels_path, f"{fn_base[:-4]}_{tile}.tif")
            
            # Extent will be N-S, and W-E boundaries
            merged_ext = {}
            merged_ext['W'], xres, _, merged_ext['N'], _, yres = [int(x) for x in self.land_cover_raster.geotransform]
            print(self.land_cover_raster.geotransform)
            print(merged_ext)
            merged_ext['E'] = merged_ext['W'] + self.ds_ncols*xres
            merged_ext['S'] = merged_ext['N'] + self.ds_nrows*yres
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
            tile_landcover = self.land_cover_raster.dataset[nrow:srow, wcol:ecol]
            print(f"Slice: {nrow}:{srow}, {wcol}:{ecol} {tile_landcover.shape}")
            if save_labels_raster:
                # Save the sliced data into a new raster
                print(f"Writing: {fn_tile} (not really)")
                # self.create_raster(fn_tile, tile_landcover, self.spatial_reference, tile_geotransform)
            
            # Slice the training mask and the NoData mask
            tile_training_mask = self.training_mask[nrow:srow, wcol:ecol]
            tile_nodata = self.no_data_arr[nrow:srow, wcol:ecol]

            # Generate features
            tile_features = self._generate_features_array(tile, fill=False)

            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_labels = os.path.join(feat_path, f"labels_{tile}.h5")

            # Save the features
            if save_features:
                print(f"Saving {tile} features...")
                # Create (large) HDF5 files to hold all features
                h5_features = h5py.File(fn_tile_features, 'w')
                h5_labels = h5py.File(fn_tile_labels, 'w')

                # Save the training and testing labels
                # h5_labels.create_dataset('all', (self.ds_nrows, self.ds_ncols), data=tile_landcover, dtype=self.land_cover_raster.dataset.dtype)
                h5_labels.create_dataset('land_cover', (self.ds_nrows, self.ds_ncols), data=tile_landcover, dtype=self.land_cover_raster.dataset.dtype)
                h5_labels.create_dataset('training_mask', (self.ds_nrows, self.ds_ncols), data=tile_training_mask, dtype=self.land_cover_raster.dataset.dtype)
                h5_labels.create_dataset('no_data_mask', (self.ds_nrows, self.ds_ncols), data=tile_nodata, dtype=self.land_cover_raster.dataset.dtype)

                for n, feature in zip(self.feat_indices, self.feat_names):
                    h5_features.create_dataset(feature, (self.ds_nrows, self.ds_ncols), data=tile_features[:,:,n])

            if by_season:
                # Aggregate features by season
                # WARNING! Requires HDF5 files to be created first!
                fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")
                self._group_features_by_season(fn_tile_features, fn_tile_feat_season)


class Plotter():

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


    def plot_array_clr(self, array: np.ndarray, fn_clr: str, **kwargs) -> None:
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

        mycmap = self.read_clr(fn_clr, zero=_zero)

        # Create a dictionary with numeric labels and colors
        colors = {}
        for k, v in enumerate(mycmap.colors):
            colors[k] = v
        # print(f'  colors={colors}')
        self.plot_array_cmap(array, colors, labels, title=_title, savefig=_savefig, dpi=_dpi)


    def read_clr(self, filename: str, zero: bool=False) -> ListedColormap:
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
    

    def plot_array_cmap(self, array: np.ndarray, colors_dict: dict, labels: list, **kwargs) -> None:
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
        norm = clrs.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = tckr.FuncFormatter(lambda x, pos: labels[norm(x)])

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



class RFLandCoverClassifierTiles(Plotter):

    """ Trains a single random forest for the mosaic, then predicts per tile """

    fmt = '%Y_%m_%d-%H_%M_%S'


    def __init__(self, feat_dataset, **kwargs):
        print("\n*** Initializing RFLandCoverClassifierTiles ***\n")
        
        self.results_suffix = kwargs.get("results_suffix", "results")

        self.feature_dataset = feat_dataset

        self.list_tiles = None
        self.y = None
        self.train_mask = None
        self.nan_mask = None
        self.x_train = None
        self.y_train = None
        self.test_mask = None
        self.x_test = None
        self.y_test = None
        self.features = None

        self.clf = None
        self.max_features = None
        self.n_estimators = None
        self.max_depth = None
        self.n_jobs = None
        self.class_weight = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.y_pred = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.overall_accuracy = None
        self.classification_report = None
        
        self.start_train = None
        self.end_train = None
        self.training_time = None

        # File names will be based on stat time
        self.start = datetime.now()
        self._configure_filenames()


    def __str__(self):
        text = self.feature_dataset.__str__()
        text += "\nLand cover classifier\n"
        text += "\n  File names:\n"
        text += f"{'Results dir':<25} {self.results_path}\n"
        text += f"{'Model':<25} {self.fn_save_model}\n"
        text += f"{'Feature importance':<25} {self.fn_save_importance}\n"
        text += f"{'Crosstab train':<25} {self.fn_save_crosstab_train}\n"
        text += f"{'Crosstab test':<25} {self.fn_save_crosstab_test}\n"
        text += f"{'Crosstab test (mask)':<25} {self.fn_save_crosstab_test_mask}\n"
        text += f"{'Confussion table':<25} {self.fn_save_conf_tbl}\n"
        text += f"{'Classification report':<25} {self.fn_save_report}\n"
        text += f"{'Predictions figure':<25} {self.fn_save_preds_fig}\n"
        text += f"{'Predictions raster':<25} {self.fn_save_preds_raster}\n"
        text += f"{'Predictions HDF5':<25} {self.fn_save_preds_h5}\n"
        # text += f"{'Parameters':<25} {self.fn_save_params}\n"
        text += f"{'Conf fig':<25} {self.fn_save_conf_fig}\n"
        text += f"{'Train error fig':<25} {self.fn_save_train_error_fig}\n"
        text += f"{'Colormap':<25} {self.fn_colormap}\n"
        
        text += "\n  Script parameters\n"
        text += f"{'Tiles':<25} {self.list_tiles}\n"
        text += f"{'Override tiles':<25} {self._override_tiles}\n"
        text += f"{'Results suffix':<25} {self.results_suffix}\n"
        text += f"{'Start':<25} {self.start}\n"
        text += f"{'CWD':<25} {cwd}\n"
        text += f"{'Format':<25} {self.fmt}\n"
        if self.x_train is not None:
            text += f"{'x_train shape':<25} {self.x_train.shape}\n"
            text += f"{'y_train shape':<25} {self.y_train.shape}\n"
            text += f"{'x_test shape':<25} {self.X.shape}\n"
            text += f"{'y_test shape':<25} {self.y.shape}\n"
        
        text += f"\n  Model parameters\n"
        text += f"{'Model type':<25} {'RandomForestClassifier'}\n"
        text += f"{'Model':<25} {self.clf}"
        text += f"{' Max features':<25} {self.max_features}\n"
        text += f"{' Estimators':<25} {self.n_estimators}\n"
        text += f"{' Max depth':<25} {self.max_depth}\n"
        text += f"{' Jobs':<25} {self.n_jobs}\n"
        text += f"{' Class weight:':<25} {self.class_weight}\n"
        text += f"{' Max features:':<25} {self.max_features}\n"
        text += f"{' OOB score':<25} {self.clf.oob_score_}\n"
        text += f"{' Training accuracy score':<25} {self.train_accuracy}\n"
        text += f"{' Testing accuracy score':<25} {self.test_accuracy}\n"
        text += f"{' Start training':<25} {self.start_train}\n"
        text += f"{' End training':<25} {self.end_train}\n"
        text += f"{' Training time':<25} {self.training_time}\n"
        
        if self.classification_report is not None:
            text += f"\nClassification report:\n"
            text += self.classification_report
        text += f"\n"
        return text
    

    def _configure_filenames(self):
        # Create directory to save results
        self.results_path = os.path.join(self.feature_dataset.get_output_dir(), self.results_suffix, self.feature_dataset.phenobased, f"{datetime.strftime(self.start, self.fmt)}")
        if not os.path.exists(self.results_path):
            print(f"\nCreating path for results: {self.results_path}")
            os.makedirs(self.results_path)
        self.fn_save_model = os.path.join(self.results_path, "rf_model.pkl")
        self.fn_save_importance = os.path.join(self.results_path, "rf_feat_importance.csv")
        self.fn_save_crosstab_train = os.path.join(self.results_path, "rf_crosstab_train.csv")
        self.fn_save_crosstab_test = os.path.join(self.results_path, "rf_crosstab_test.csv")
        self.fn_save_crosstab_test_mask = self.fn_save_crosstab_test[:-4] + '_mask.csv'
        self.fn_save_conf_tbl = os.path.join(self.results_path, "rf_confussion_table.csv")
        self.fn_save_report = os.path.join(self.results_path, "rf_report.txt")
        self.fn_save_preds_fig = os.path.join(self.results_path, "rf_predictions.png")
        self.fn_save_preds_raster = os.path.join(self.results_path, "rf_predictions.tif")
        self.fn_save_preds_h5 = os.path.join(self.results_path, "rf_predictions.h5")
        # self.fn_save_params = os.path.join(self.results_path, "rf_parameters.csv")
        self.fn_save_conf_fig = self.fn_save_conf_tbl[:-4] + '.png'
        self.fn_save_train_error_fig = os.path.join(self.results_path, "rf_training_error.png")

        self.fn_colormap = self.feature_dataset.get_output_dir() + 'parameters/qgis_cmap_landcover_CBR_viri_grp11.clr'


    def _load_mosaic_features(self):
        """Loads features and labels from HDF5 files.

        Features dataset is an assembled mosaic from small tiles.
        Labels should be generated beforehand with the dimensions to match the features mosaic.
        """
        start_loading = datetime.now()
        assert self.feature_dataset.land_cover_raster.nrows is not None, f"Value not set for nrows"
        assert self.feature_dataset.land_cover_raster.ncols is not None, f"Value not set for ncols"
        assert self.feature_dataset.feat_names is not None, f"Value not set for features"

        # Read the labels
        print(f"\nReading labels: {self.feature_dataset.fn_feature_labels}")
        with h5py.File(self.feature_dataset.fn_feature_labels, 'r') as h5_labels:
            # self.y = h5_labels['all'][:]
            self.y = h5_labels['land_cover'][:]
            self.train_mask = h5_labels['training_mask'][:]
            self.nan_mask = h5_labels['no_data_mask'][:]
        print("Flattening labels and masks...")
        # self.y = self.feature_dataset.land_cover_raster.dataset.flatten()
        self.y = self.y.flatten()  #TODO: check if this is the same as the line above!
        self.train_mask = self.train_mask.flatten()
        self.nan_mask = self.nan_mask.flatten()

        # Read the feature list names
        self.features = self.feature_dataset.feat_names
        if self.feature_dataset.feat_names_season is not None:
            self.features = self.feature_dataset.feat_names_season

        # Read features into a 2D array, read features from each tile one-by-one
        self.X = np.zeros((self.feature_dataset.land_cover_raster.nrows*self.feature_dataset.land_cover_raster.ncols, len(self.features)),
                           dtype=self.feature_dataset.land_cover_raster.dtype)
        print(f"  X (empty)  shape={self.X.shape}, size={(self.X.size * self.X.itemsize)//(1000*1000*1000)} GiB")
        print(f"  y          shape={self.y.shape}, size={(self.y.size * self.y.itemsize)//(1000*1000)} MiB {self.y.size} {self.y.itemsize} {self.y.dtype}")
        print(f"  train_mask shape={self.train_mask.shape}, size={(self.train_mask.size * self.train_mask.itemsize)//(1000*1000)} MiB {self.train_mask.size} {self.train_mask.itemsize} {self.train_mask.dtype}")
        print(f"  nan_mask   shape={self.nan_mask.shape}, size={(self.nan_mask.size * self.nan_mask.itemsize)//(1000*1000)} MiB")

        # Read features from tiles, create a mosaic of features, and reshape it into 2D dataset of features
        # print(self.feature_dataset.tiles_slice)
        tiles_per_row = self.feature_dataset.land_cover_raster.ncols / self.feature_dataset.ds_ncols
        for i, tile in enumerate(self.feature_dataset.tiles):
            print(f"\n== Reading features for tile {tile} ({i+1}/{len(self.feature_dataset.tiles)}) ==")
            feat_path = os.path.join(self.feature_dataset.get_output_dir(), self.feature_dataset.features_suffix, self.feature_dataset.phenobased, tile)
            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")

            # print(fn_tile_features)
            # print(fn_tile_feat_season)

            assert os.path.isfile(fn_tile_feat_season) or os.path.isfile(fn_tile_features), "ERROR: features files not found!"

            if os.path.isfile(fn_tile_feat_season):
                # Look for seasonal features first
                print(f"--Found seasonal features: {fn_tile_feat_season}.")
                fn_tile_features = fn_tile_feat_season  # point to features by season
            elif os.path.isfile(fn_tile_features):
                print(f"--Found monthly features: {fn_tile_features}")
            # print(self.features)

            # Get rows and columns to insert features
            tile_row = self.feature_dataset.tiles_slice[tile]['N']
            tile_col = self.feature_dataset.tiles_slice[tile]['W']

            # Account for number of tiles (or steps) per row/column
            row_steps = tile_row // self.feature_dataset.ds_nrows
            col_steps = tile_col // self.feature_dataset.ds_ncols

            # Read the features, for re of tiles_per_row according to current row
            tile_start = int((tiles_per_row * row_steps + col_steps) * (self.feature_dataset.ds_nrows*self.feature_dataset.ds_ncols))

            # print(f"--tile row={tile_row}")
            print(f"--Reading the features from: {fn_tile_features}")
            feat_array = np.empty((self.feature_dataset.ds_nrows, self.feature_dataset.ds_ncols, len(self.features)), dtype=self.feature_dataset.land_cover_raster.dtype)
            with h5py.File(fn_tile_features, 'r') as h5_features:
                print(f"Features in file: {len(list(h5_features.keys()))}, feature names: {len(self.features)} ")
                # Get the data from the HDF5 files
                for i, feature in enumerate(self.features):
                    feat_array[:,:,i] = h5_features[feature][:]

            # Transform into a 2D-array
            # Insert tile features in the right position of the 2-D array
            print(f"--Inserting dataset into 2D array ({self.feature_dataset.ds_nrows}x{self.feature_dataset.ds_ncols})...")
            for row in range(self.feature_dataset.ds_nrows):
                for col in range(self.feature_dataset.ds_ncols):
                    # Calculate the right position to insert the datset
                    insert_row = tile_start + row*self.feature_dataset.ds_nrows + col
                    if row == 0 and col == 0:
                        print(f"--Starting at row: {insert_row}")
                    self.X[insert_row, :] = feat_array[row, col, :]
            print(f"--Finished at row: {insert_row}")
        
        # Features read successfully
        print("Done reading features.\n")

        print("Creating training and testing datasets...")
        print(f"  train_mask: shape={str(self.train_mask.shape):<20} size={(self.train_mask.size * self.train_mask.itemsize)//(1000*1000):<4} MiB")
        self.x_train = self.X[self.train_mask > 0]
        self.y_train = self.y[self.train_mask > 0]

        # # TODO: check if this is neccesary, already done when creating dataset?
        # # Create a TESTING MASK: Select on the valid region only (discard NoData pixels)
        # self.test_mask = np.logical_and(self.train_mask == 0, self.nan_mask == 1)
        # self.x_test = self.X[self.test_mask]
        # self.y_test = self.y[self.test_mask]

        print(f'\n  x_train shape={str(self.x_train.shape):<20} size={(self.x_train.size * self.x_train.itemsize)//(1000*1000):<4} MiB')
        print(f'  y_train shape={str(self.y_train.shape):<20} size={(self.y_train.size * self.y_train.itemsize)//(1000*1000):<4} MiB')
        # print(f'  x_test  shape={str(self.x_test.shape):<20} size={(self.x_test.size * self.x_test.itemsize)//(1000*1000*1000):<4} GiB')
        # print(f'  y_test  shape={str(self.y_test.shape):<20} size={(self.y_test.size * self.y_test.itemsize)//(1000*1000):<4} MiB')
        print(f'  X       shape={str(self.X.shape):<20} size={(self.X.size * self.X.itemsize)//(1000*1000*1000):<4} GiB')
        print(f'  y       shape={str(self.y.shape):<20} size={(self.y.size * self.y.itemsize)//(1000*1000):<4} MiB')

        # Save train datasets after sample mask (train_mask), has been applied, mainly for debugging
        print("Saving training datasets")
        fn_train = os.path.join(self.results_path, "rf_train_dataset.h5")
        with h5py.File(fn_train, 'w') as h5_train:
            h5_train.create_dataset("x_train", self.x_train.shape, data=self.x_train)
            h5_train.create_dataset("y_train", self.y_train.shape, data=self.y_train)

        end_loading = datetime.now()
        loading_time = end_loading - start_loading
        print(f'{end_loading}: creating dataset in {loading_time}.')


    def rf_train_optim(self, **kwargs):
        """Creates the random forest model and searchs optimum parameters, then fits the training dataset"""
        _min_estimators = kwargs.get("min_estimators", 200)
        _max_estimators = kwargs.get("max_estimators", 400)
        _step_estimators = kwargs.get("step_estimators", 200)
        _dpi = kwargs.get("dpi", 300)

        # Create and read the features mosaic dataset and the labels dataset
        self._load_mosaic_features()

        print(f"\n*** Starting Random Forest training ***")

        RANDOM_STATE = 42  # same for reproducibility
        JOBS = 64

        ensemble_clfs = [
            (
                "RandomForestClassifier, max_features='sqrt'",
                RandomForestClassifier(
                    warm_start=True,
                    oob_score=True,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    verbose=1,
                    n_jobs=JOBS,
                ),
            ),
            (
                "RandomForestClassifier, max_features='log2'",
                RandomForestClassifier(
                    warm_start=True,
                    max_features="log2",
                    oob_score=True,
                    random_state=RANDOM_STATE,
                    verbose=1,
                    n_jobs=JOBS,
                ),
            ),
            (
                "RandomForestClassifier, max_features=None",
                RandomForestClassifier(
                    warm_start=True,
                    max_features=None,
                    oob_score=True,
                    random_state=RANDOM_STATE,
                    verbose=1,
                    n_jobs=JOBS,
                ),
            ),
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Explore a range of `n_estimators` values in each `max_features`.
        for label, clf in ensemble_clfs:
            print(f"\n*** Processing: {label} ***")
            for i in range(_min_estimators, _max_estimators + 1, _step_estimators):
                print(f"\nFitting model with {i}/{_max_estimators} estimators @ {_step_estimators}\n")

                clf.set_params(n_estimators=i)
                clf.fit(self.x_train, self.y_train)

                # Record the OOB error for each `n_estimators=i` setting.
                print(f"OOB score={clf.oob_score_}")
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(_min_estimators, _max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.savefig(self.fn_save_train_error_fig, bbox_inches='tight', dpi=_dpi)
        # plt.show()


    def rf_train(self, **kwargs):
        """Creates the random forest model and fits the training dataset"""
        self.n_estimators = kwargs.get("n_estimators", 250)
        self.max_depth = kwargs.get("max_depth", None)
        self.n_jobs = kwargs.get("n_jobs", 64)
        self.class_weight = kwargs.get("class_weight", None)
        self.max_features = kwargs.get("max_featuers", None)
        # self.override_tiles = kwargs.get("override_tiles", None)
        _save_model = kwargs.get("save_model", False)

        # Create and read the features mosaic dataset and the labels dataset
        self._load_mosaic_features()

        print(f"\n*** Random Forest training ***\n")

        self.start_train = datetime.now()
        print(f'{self.start_train}: starting Random Forest training')

        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                    oob_score=True,
                                    max_features=self.max_features,
                                    max_depth=self.max_depth,
                                    n_jobs=self.n_jobs,
                                    class_weight=self.class_weight,
                                    verbose=1)
        print(f"Fitting model with max_features {self.max_features} and {self.n_estimators} estimators.")

        # IMPORTANT: This replaces the initial model by the trained model!
        self.clf = self.clf.fit(self.x_train, self.y_train)

        # Save trained model
        if _save_model:
            print("Saving trained model...")
            with open(self.fn_save_model, 'wb') as f:
                pickle.dump(self.clf, f)
        
        print(f'  OOB score: {self.clf.oob_score_ * 100:0.2f}%')

        # Calculate feature importance
        feat_list = []
        feat_imp = []
        for feat, imp in zip(self.features, self.clf.feature_importances_):
            feat_list.append(feat)
            feat_imp.append(imp)
        feat_importance = pd.DataFrame({'Feature': feat_list, 'Importance': feat_imp})
        feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
        print("Feature importance: ")
        print(feat_importance.to_string())
        feat_importance.to_csv(self.fn_save_importance)

        self.end_train = datetime.now()
        self.training_time = self.end_train - self.start_train
        print(f'{self.end_train}: training finished in {self.training_time}.')


    def predict_training(self):
        start_pred_train = datetime.now()
        print(f'\n{start_pred_train}: starting predictions for training dataset.')

        # Predict the train dataset
        self.y_pred_train = self.clf.predict(self.x_train)

        # A crosstabulation to see class confusion for TRAINING
        df_tr = pd.DataFrame({'truth': self.y_train, 'predict': self.y_pred_train})
        crosstab_tr = pd.crosstab(df_tr['truth'], df_tr['predict'], margins=True)
        crosstab_tr.to_csv(self.fn_save_crosstab_train)

        self.train_accuracy = accuracy_score(self.y_train, self.y_pred_train)

        print(f'  Training predictions shape:', self.y_pred_train.shape)
        print(f'  Training accuracy score: {self.train_accuracy}')
        print(f"  Training predictions values: {np.unique(self.y_pred_train)}")
        
        del(self.y_pred_train)
        del(self.x_train)
        gc.collect()

        end_pred_train = datetime.now()
        pred_train_elapsed = end_pred_train - start_pred_train
        print(f'{end_pred_train}: predictions for training dataset finished in {pred_train_elapsed}.')


    def predict_testing(self):
        """Predictions for the testing dataset"""
        start_pred_test = datetime.now()
        print(f'\n{start_pred_test}: starting predictions for testing dataset.')

        self.y_pred_test = self.clf.predict(self.x_test)

        # A crosstabulation to see class confusion for TESTING (MASKED)
        df_ts = pd.DataFrame({'truth': self.y_test, 'predict': self.y_pred_test})
        crosstab_ts = pd.crosstab(df_ts['truth'], df_ts['predict'], margins=True)
        crosstab_ts.to_csv(self.fn_save_crosstab_test_mask)

        self.test_accuracy = accuracy_score(self.y_test, self.y_pred_test)

        print(f'  Testing predictions shape:', self.y_pred_test.shape)
        print(f'  Testing accuracy score: {self.test_accuracy}')
        print(f'  Testing predictions values: {np.unique(self.y_pred_test)}')

        cm = confusion_matrix(self.y_test, self.y_pred_test)
        with open(self.fn_save_conf_tbl, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for single_row in cm:
                writer.writerow(single_row)

        self.classification_report = classification_report(self.y_test, self.y_pred_test, )
        print('  Classification report')
        print(self.classification_report)
        # with open(self.fn_save_report, 'w') as f:
        #     f.write(report)

        del(self.y_pred_test)
        del(self.x_test)
        gc.collect()
        
        end_pred_test = datetime.now()
        pred_test_elapsed = end_pred_test - start_pred_test
        print(f'{end_pred_test}: predictions for testing dataset finished in {pred_test_elapsed}.')


    def predict_all(self, **kwargs):
        _save_plot = kwargs.get("save_plot", False)
        _save_raster = kwargs.get("save_raster", False)

        start_pred_all = datetime.now()
        print(f'\n{start_pred_all}: starting predictions for complete dataset.')

        self.y_pred = self.clf.predict(self.X)

        # A crosstabulation to see class confusion for TESTING (COMPLETE MAP)
        df = pd.DataFrame({'truth': self.y, 'predict': self.y_pred})
        crosstab = pd.crosstab(df['truth'], df['predict'], margins=True)
        crosstab.to_csv(self.fn_save_crosstab_test)

        self.overall_accuracy = accuracy_score(self.y, self.y_pred)

        print(f'  Overall predictions shape:', self.y_pred.shape)
        print(f'  Overall accuracy score: {self.test_accuracy}')

        # Reshape the classification map into a 2D array again to show as a map
        self.y_pred = self.y_pred.reshape((self.feature_dataset.land_cover_raster.nrows,
                                           self.feature_dataset.land_cover_raster.ncols))

        print(f'  Predictions (re)shape:', self.y_pred.shape)
        print(f'  Predictions values: {np.unique(self.y_pred)}')

        # Plot the land cover map of the predictions for y and the whole area
        # zero=True, zeros removed with mask?
        if _save_plot:
            self.plot_array_clr(self.y_pred, self.fn_colormap, savefig=self.fn_save_preds_fig, zero=True)

        # Save predicted land cover classes into a GeoTIFF
        if _save_raster:
            self.feature_dataset.land_cover_raster.create_raster(self.fn_save_preds_raster,
                                                                self.y_pred,
                                                                self.feature_dataset.land_cover_raster.spatial_reference,
                                                                self.feature_dataset.land_cover_raster.geotransform)

        # Save predicted land cover classes into a HDF5 file
        with h5py.File(self.fn_save_preds_h5, 'w') as h5_preds:
            h5_preds.create_dataset("predictions", self.y_pred.shape, data=self.y_pred)
        
        del(self.y_pred)
        del(self.y)
        del(self.X)
        gc.collect()

        end_pred_all = datetime.now()
        pred_all_elapsed = end_pred_all - start_pred_all
        print(f'{end_pred_all}: predictions for complete dataset finished in {pred_all_elapsed}.')


    def predict_all_mosaic(self, **kwargs):
        """Predictions fot the entire dataset using a mosaic approach

        :param list override_tiles: predict only for tiles specified in the list
        :param str model: file name of a previously trained model (in Pickle format).
        """
        start_pred_test = datetime.now()
        self._override_tiles = kwargs.get("override_tiles", None)
        _pretrained_model = kwargs.get("model", None)

        # Read the feature list names
        self.features = self.feature_dataset.feat_names
        if self.feature_dataset.feat_names_season is not None:
            self.features = self.feature_dataset.feat_names_season

        # If specified trained model
        if _pretrained_model is not None:
            # Load previously trained model
            print(f"\n==Loading pretrained model: {_pretrained_model}==")
            with open(_pretrained_model, 'rb') as model:
                self.clf = pickle.load(model)

        print(f"\n*** Predict for complete dataset ***")
        print(f'\n{start_pred_test}: starting (mosaic) predictions for complete dataset.')

        # Prepare 2D mosaic to save predictions (same shape as land cover raster)
        self.y_pred = np.zeros(self.feature_dataset.land_cover_raster.shape,
                                    dtype=self.feature_dataset.land_cover_raster.dtype)
        mosaic_nan_mask = np.zeros(self.feature_dataset.land_cover_raster.shape,
                                    dtype=self.feature_dataset.land_cover_raster.dtype)

        # Read features from tiles, create a mosaic of features, and reshape it into 2D dataset of features
        tiles_per_row = self.feature_dataset.land_cover_raster.ncols / self.feature_dataset.ds_ncols
        rows_per_tile = self.feature_dataset.ds_ncols * self.feature_dataset.ds_nrows
        print(f"tiles_per_row={tiles_per_row} (tiles in mosaic), rows_per_tile={rows_per_tile}")

        # Predict by reading the features of each tile from its corresponding HDF5 file
        for i, tile in enumerate(self.feature_dataset.tiles):
            # User can override the list of tiles to make predictions
            if (self._override_tiles is not None) and (tile not in self._override_tiles):
                print(f"Skipping tile {tile} (overrided by user).")
                i += 1
                continue
            print(f"\n== Making predictions for tile {tile} ({i+1}/{len(self.feature_dataset.tiles)}) ==")

            # Look for either monthly or seasonal feature HDF5 files for current tile
            feat_path = os.path.join(self.feature_dataset.get_output_dir(), self.feature_dataset.features_suffix, self.feature_dataset.phenobased, tile)
            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")
            # print(fn_tile_features)
            # print(fn_tile_feat_season)

            # Check if either monthly or seasonal feature files were found
            assert os.path.isfile(fn_tile_feat_season) or os.path.isfile(fn_tile_features), "ERROR: features files not found!"
            # If both seasonal and monthly features found, seasonal are preferred
            if os.path.isfile(fn_tile_feat_season):
                # Look for seasonal features first
                print(f"--Found seasonal features: {fn_tile_feat_season}.")
                fn_tile_features = fn_tile_feat_season  # point to features by season
            elif os.path.isfile(fn_tile_features):
                print(f"--Found monthly features: {fn_tile_features}")
            # print(self.features)

            print(f"--Reading tile features from: {fn_tile_features}")
            # feat_array = np.empty((self.feature_dataset.ds_nrows, self.feature_dataset.ds_ncols, len(self.features)), dtype=self.feature_dataset.land_cover_raster.dtype)
            feat_array = np.empty((self.feature_dataset.ds_nrows, self.feature_dataset.ds_ncols, len(self.features)))
            with h5py.File(fn_tile_features, 'r') as h5_features:
                print(f"  Features in file: {len(list(h5_features.keys()))}, feature names: {len(self.features)} ")
                # Get the data from the HDF5 files
                for i, feature in enumerate(self.features):
                    feat_array[:,:,i] = h5_features[feature][:]

            # Prepare 2D array to save tile features
            X_tile = feat_array.reshape(self.feature_dataset.ds_nrows*self.feature_dataset.ds_ncols, len(self.features))

            # Read labels and no_data mask
            fn_labels_tile = os.path.join(self.feature_dataset.get_output_dir(), #self.land_cover_raster.output_dir,
                                       self.feature_dataset.features_suffix,
                                       self.feature_dataset.phenobased,
                                       tile,
                                       f"labels_{tile}.h5")
            print(f"--Reading tile labels from: {fn_labels_tile}")
            with h5py.File(fn_labels_tile, 'r') as h5_labels_tile:
                # y_tile = h5_labels_tile['all'][:]
                y_tile = h5_labels_tile['land_cover'][:]
                y_tile_nd = h5_labels_tile['no_data_mask'][:]
            # y_tile_nd = y_tile_nd.flatten()
            print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape}")

            # Predict for tile
            y_pred_tile = self.clf.predict(X_tile)
            print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape} y_pred_tile={y_pred_tile.shape}")

            # Reshape list of predictions as 2D image and filter pixels with the no_data_mask
            y_pred_img = y_pred_tile.reshape((self.feature_dataset.ds_nrows,
                                               self.feature_dataset.ds_ncols))
            y_pred_img = np.where(y_tile_nd == 1, y_pred_img, 0)
            
            print("Inserting tile predictions into mosaic")
            # Get rows and columns to insert tile predictions into mosaic
            tile_row = self.feature_dataset.tiles_slice[tile]['N']
            tile_col = self.feature_dataset.tiles_slice[tile]['W']
            self.y_pred[tile_row:tile_row+self.feature_dataset.ds_nrows, tile_col:tile_col+self.feature_dataset.ds_nrows] = y_pred_img
            mosaic_nan_mask[tile_row:tile_row+self.feature_dataset.ds_ncols, tile_col:tile_col+self.feature_dataset.ds_ncols] = y_tile_nd

            # Save predicted land cover classes into a HDF5 file (for debugging purposes)
            print("Saving tile predictions (as HDF5 file)")
            with h5py.File(self.fn_save_preds_h5[:-3] + f'_{tile}.h5', 'w') as h5_preds_tile:
                h5_preds_tile.create_dataset(f"{tile}_feat", feat_array.shape, data=feat_array)
                # h5_preds_tile.create_dataset(f"{tile}_x", X_tile.shape, data=X_tile)  # 1D list
                # h5_preds_tile.create_dataset(f"{tile}_y", y_pred_tile.shape, data=y_pred_tile)  # 1D list
                h5_preds_tile.create_dataset(f"{tile}_ypred", y_pred_img.shape, data=y_pred_img)

        print("\nFinished tile predictions.")
        print("Saving the mosaic predictions (raster and h5).")
        # Save predictions into a raster (and no_data_mask for debugging)
        self.feature_dataset.land_cover_raster.create_raster(self.fn_save_preds_raster,
                                                             self.y_pred,
                                                             self.feature_dataset.land_cover_raster.spatial_reference,
                                                             self.feature_dataset.land_cover_raster.geotransform)
        self.feature_dataset.land_cover_raster.create_raster(self.fn_save_preds_raster[:-4] + "_gen_nan_mask.tif",
                                                             mosaic_nan_mask,
                                                             self.feature_dataset.land_cover_raster.spatial_reference,
                                                             self.feature_dataset.land_cover_raster.geotransform)
        # Save predicted land cover classes into a HDF5 file
        with h5py.File(self.fn_save_preds_h5, 'w') as h5_preds:
            h5_preds.create_dataset("predictions", self.y_pred.shape, data=self.y_pred)

        end_pred_test = datetime.now()
        pred_test_elapsed = end_pred_test - start_pred_test
        print(f'{end_pred_test}: predictions for testing dataset finished in {pred_test_elapsed}.')


    def save_report(self):
        """Saves a text file with all the model training and prediction parameters """
        with open(self.fn_save_report, 'w') as f:
            f.write(self.__str__())


class LandCoverClassifier(Plotter):

    """ Trains a random forest algorithm per tile and makes predictions """

    fmt = '%Y_%m_%d-%H_%M_%S'
    
    def __init__(self, feat_dataset, **kwargs):
        print("Initializing LandCoverClassifier")
        
        self.results_suffix = kwargs.get("results_suffix", "results")

        self.feature_dataset = feat_dataset

        self.list_tiles = None

    def __str__(self):
        text = self.feature_dataset.__str__()
        text += "\nLand cover classifier"
        text += f"{'Tiles':>25} {self.list_tiles}\n"
        text += f"{'Results suffix':>25} {self.results_suffix}\n"
        return text
    
    def rf_classify(self, tiles=None):

        start = datetime.now()

        if tiles is not None:
            print("WARNING!: Overriding list of tiles!")
            self.list_tiles = tiles
        else:
            self.list_tiles = self.feature_dataset.tiles
        
        # If no tiles generate a single array of pre
        if self.list_tiles is None:
            self.list_tiles = ['']
            print("Running  single algorithm")

        # Create directory to save results
        results_path = os.path.join(self.feature_dataset.get_output_dir(), self.results_suffix, self.feature_dataset.phenobased, f"{datetime.strftime(start, self.fmt)}")
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
            
            fn_colormap = self.feature_dataset.get_output_dir() + 'parameters/qgis_cmap_landcover_CBR_viri_grp11.clr'
            
            # Look for feature files in each tile
            feat_path = os.path.join(self.feature_dataset.get_output_dir(), self.feature_dataset.features_suffix, self.feature_dataset.phenobased, tile)
            fn_tile_labels = os.path.join(feat_path, f"labels_{tile}.h5")
            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")

            # print(fn_tile_labels)
            # print(fn_tile_features)
            # print(fn_tile_feat_season)

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
            
            # TODO: Change to uint8? Requires generating all NDVI and EVI datasets!!!
            # x_train = np.empty((self.nrows, self.ncols, len(self.features)), dtype=np.int16)
            # y = np.empty((self.nrows, self.ncols), dtype=np.int8)
            # train_mask = np.empty((self.nrows, self.ncols), dtype=np.int8)
            # nan_mask = np.empty((self.nrows, self.ncols), dtype=np.int8)

            # Read the labels
            print("Reading labels...")
            with h5py.File(fn_tile_labels, 'r') as h5_labels:
                # self.y = h5_labels['all'][:]
                self.y = h5_labels['land_cover'][:]
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

            print(f'  --x_train shape={self.x_train.shape} {self.x_train.size}')
            print(f'  --y_train shape={self.y_train.shape} {self.y_train.size}')
            print(f'  --x_test shape={self.x_test.shape}')
            print(f'  --y_test shape={self.y_test.shape}')
            print(f'  --X shape={self.X.shape}')
            print(f'  --y shape={self.y.shape}')

            if self.x_train.size == 0 or self.y_train.size == 0:
                print(f"Skipping empty tile: {tile}")
                continue

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
            self.plot_array_clr(y_pred, fn_colormap, savefig=fn_save_preds_fig, zero=True)  # zero=True, zeros removed with mask?

            # Save predicted land cover classes into a GeoTIFF
            # Get geotransform from raster label
            fn_tile_labels_raster = os.path.join(self.feature_dataset.get_output_dir(), self.feature_dataset.labels_suffix, tile, f"usv250s7cw2018_ROI2full_{tile}.tif")
            _, _, tile_geotransform, tile_spatial_ref = self.feature_dataset.land_cover_raster.open_raster(fn_tile_labels_raster)
            self.feature_dataset.land_cover_raster.create_raster(fn_save_preds_raster, y_pred, tile_spatial_ref, tile_geotransform)

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


if __name__ == '__main__':

    start = datetime.now()

    # *** Testing code ****
    cwd = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/"
    dir_bands = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/02_STATS/"
    dir_pheno = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/03_PHENO/"
    dir_ancillary = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/data/ancillary/"
    ancillary_dict = {101: ["ag_roi2.tif"], 102: ["roads_roi2.tif", "urban_roi2.tif"]}
    fn_tiles = os.path.join(cwd, 'parameters/tiles')

    fn_landcover = os.path.join(cwd, "data/inegi/usv250s7cw2018_ROI2full_ancillary.tif")

    # This will incorporate the ancillary data rasters, and carry out the stratified random sampling
    raster = LandCoverRaster(fn_landcover, cwd)
    # raster.incorporate_ancillary(ancillary_dir=dir_ancillary,
    #                              ancillary_dict=ancillary_dict)
    raster.configure_sampling_files()
    # raster.sample(max_trials=1e6)
    # print(raster)

    features = FeaturesDataset(raster, dir_bands, dir_pheno, file_tiles=fn_tiles)
    # create features (this is time consuming!)
    # features.create_features_dataset(by_season=True)
    features.create_labels_dataset()  # run only once
    # features.create_tile_dataset(by_season=True) # use this instead the two lines above
    # ...or read feature parameters from existing datasets
    features.read_params_from_features()
    # print(features)
    

    # # CASE 1: One RF model per tile
    # lcc = LandCoverClassifier(features)
    # lcc.rf_classify()
    # print(lcc)

    # # CASE 2: Single RF for complete area
    # # CASE 2.1: Train and predict for the entire ROI. WARNING: This takes time!
    # lcc = RFLandCoverClassifierTiles(features)
    # lcc.rf_train(save_model=True)
    # lcc.predict_all_mosaic()
    # lcc.save_report()

    # CASE 2.2: Use a previously trained model to predict.
    lcc = RFLandCoverClassifierTiles(features)
    # Make predictions using a previously trained model
    trained_model = os.path.join(cwd, 'results/NDVI/2023_08_23-18_01_09/', 'rf_model.pkl')
    lcc.predict_all_mosaic(override_tiles=['h19v25'], model=trained_model)
    lcc.save_report()

    # lcc.predict_training()
    # lcc.predict_testing()
    # lcc.predict_all(save_plot=True, save_raster=True)
    # lcc.save_results_report()
    # print(lcc)

    print(f"\nEverything completed on: {datetime.now() - start}. Bye ;-)")
