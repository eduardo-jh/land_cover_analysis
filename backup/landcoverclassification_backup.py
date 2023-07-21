#!/usr/bin/env python
# coding: utf-8

""" Classes for land cover classification
"""
import os
import gc
import csv
import random
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from scipy import stats
from osgeo import gdal
from osgeo import osr
from pyhdf.SD import SD, SDC

plt.style.use('ggplot')  # R-like plots


class LandCoverRaster():
    """ Reads land cover from a raster file (GeoTIFF) """

    def __init__(self, fn_landcover, output_dir):

        assert os.path.isfile(fn_landcover), f"File not found: {fn_landcover}"
        assert os.path.isdir(output_dir), f"Directory not found: {output_dir}"

        self.output_dir = output_dir

        self.fn_landcover = fn_landcover
        self.dataset = None
        self.geotransform = None
        self.nodata = None
        self.spatial_reference = None
        self.nrows = None
        self.ncols = None
        self.data_type = None
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

        print("Initializing land cover raster... done.")


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
        text += f"{'Data type':<25} {self.data_type}\n"
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
        
        return text


    def read_land_cover(self):
        """ Populates the raster data and parameters"""
        self.dataset, self.nodata, self.geotransform, self.spatial_reference = self.open_raster(self.fn_landcover)
        self.nrows = self.dataset.shape[0]
        self.ncols = self.dataset.shape[1]
        self.nodata = int(self.nodata)
        self.data_type = self.dataset.dtype
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
        print(type(self.spatial_reference), self.spatial_reference)
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


    def sample(self, **kwargs):
        """ Stratified random sampling

        :param float train_percent: default training-testing proportion is 80-20%
        :param int win_size: default is sampling a window of 7x7 pixels
        :param int max_trials: max of attempts to fill the sample size
        """
        self.fig_frequencies = kwargs.get("fig_frequencies", "class_fequencies.png")
        self.train_percent = kwargs.get("train_percent", 0.2)
        self.window_size = kwargs.get("window_size", 7)
        self.max_trials = int(kwargs.get("max_trials", 2e5))
        self.sampling_suffix = kwargs.get("sampling_suffix", "sampling")
        self.fn_training_mask = kwargs.get("training_mask", "training_mask.tif")
        self.fn_training_labels = kwargs.get("training_labels", "training_labels.tif")
        self.fn_sample_sizes = kwargs.get("sample_sizes", "dataset_sample_sizes.csv")

        sampling_dir = os.path.join(self.output_dir, self.sampling_suffix)
        if not os.path.exists(sampling_dir):
                print(f"\nCreating new path: {sampling_dir}")
                os.makedirs(sampling_dir)
        
        self.fn_training_mask = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_mask)
        self.fn_training_labels = os.path.join(self.output_dir, self.sampling_suffix, self.fn_training_labels)
        self.fig_frequencies = os.path.join(self.output_dir, self.sampling_suffix, self.fig_frequencies)
        self.fn_sample_sizes = os.path.join(self.output_dir, self.sampling_suffix, self.fn_sample_sizes)

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



class FeaturesDataset(LandCoverRaster):
    """ Reads spectral bands and phenology HDF4 files and creates HDF5 feature files """

    FEAT_OFFSET = 5  # not part of feature name
    PHENO_OFFSET = 4
    BAND_SPECTRAL = 0
    BAND_VI = 1
    BAND_PHENOLOGY_1 = 2
    BAND_PHENOLOGY_2 = 3
    

    def __init__(self, fn_landcover, bands_dir, pheno_dir, output_dir, **kwargs):
        self.features_dir = kwargs.get("features_dir", "features")
        self.fn_tiles = kwargs.get("file_tiles", None)
        self.phenobased = kwargs.get("phenobased", "NDVI")
        self.feat_list = kwargs.get("feat_list", None)
        self.fn_training_mask = kwargs.get("training_mask", "")

        assert os.path.isdir(bands_dir), f"Directory not found: {bands_dir}"
        assert os.path.isdir(pheno_dir), f"Directory not found: {pheno_dir}"
        assert os.path.isdir(output_dir), f"Directory not found: {output_dir}"
        assert os.path.isfile(fn_landcover), f"File not found: {fn_landcover}"
        LandCoverRaster(fn_landcover, output_dir)

        if self.fn_training_mask != "":
            assert os.path.isfile(self.fn_training_mask), f"File not found: {self.fn_training_mask}"

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

        self.ds_nrows = None
        self.ds_ncols = None
        self.tiles = None
        self.tiles_extent = None
        self.training_mask = None
        self.no_data_arr = None

        print("Initializing directories to parse HDF... done!")


    def __str__(self):
        text = "\n"
        text += LandCoverRaster.__str__()
        text += "\nFeatures dataset:\n"
        text += f"{'Spectral bands directory':>25} {self.bands_dir}\n"
        text += f"{'Phenology directory':>25} {self.pheno_dir}\n"
        text += f"{'Output directory':>25} {self.output_dir}\n"
        text += f"{'Features directory':>25} {self.features_dir}\n"
        text += f"{'File with tiles':>25} {self.fn_tiles}\n"
        text += f"{'Phenology based':>25} {self.phenobased}\n"
        text += f"{'List of features':>25} {self.feat_list}\n"
        text += f"{'Training mask file':>25} {self.fn_training_mask}\n"
        text += f"{'Feature indices':>25} {self.feat_indices}\n"
        text += f"{'Feature names':>25} {self.feat_names}\n"
        text += f"{'Feature types':>25} {self.feat_type}\n"
        text += f"{'Months':>25} {self.months}\n"
        text += f"{'Variables':>25} {self.variables}\n"
        text += f"{'Phenology':>25} {self.phenology}\n"
        text += f"{'Phenology2':>25} {self.phenology2}\n"
        text += f"{'Fill':>25} {self.fill}\n"
        text += f"{'Standardize':>25} {self.standardize}\n"
        text += f"{'Normalize':>25} {self.normalize}\n"
        text += f"{'Dataset rows':>25} {self.ds_nrows}\n"
        text += f"{'Dataset columns':>25} {self.ds_ncols}\n"
        text += f"{'Tiles':>25} {self.tiles}\n"

        return text
    

    def read_from_hdf(filename: str, var: str, as_int16: bool=False) -> np.ndarray:
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

        if as_int16:
            data_arr = np.round(data_arr).astype(np.int16)

        return data_arr
    

    def get_band(self, feature_name: str) -> str:
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
    

    def fix_annual_phenology(self, data: np.ndarray) -> np.ndarray:
        """Fix phenology values larger than 366 and returns the fixed dataset"""
        MAX_ITERS = 10
        iter = 0
        while np.max(data) > 366 or iter >= MAX_ITERS:
            data = np.where(data > 366, data-365, data)
            iter += 1
        return data
    

    def normalize_dataset(self, ds: np.ndarray) -> np.ndarray:
        """ Normalize a dataset with min-max feature scaling into range [0,1] """
        _min = np.nanmin(ds)
        _max = np.nanmax(ds)
        return (ds - _min) / (_max - _min)


    def standardize_dataset(self, ds: np.ndarray) -> np.ndarray:
        """ Standarize a dataset into range [-1, 1] """
        avg = np.nanmean(ds)
        std = np.nanstd(ds)
        return (ds - avg) / std
    

    def fill_with_mean(self, dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def fill_with_mode(self, data: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def fill_with_int_mean(self, dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
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


    def fill_season(self, sos: np.ndarray, eos: np.ndarray, los: np.ndarray, min_value: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


    def generate_feature_list(self):
        """Generates the list of features (names and indices) to be used"""

        self.feat_indices = []
        self.feat_names = []
        self.feat_type = []

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
        
        # Identify the type of data
        for feat in self.feat_names:
            band = self.get_band(feat)
            band_type = self.BAND_SPECTRAL
            if band in ["NDVI", "EVI", "EVI2"]:
                band_type = self.BAND_VI
            elif band in self.phenology:
                band_type = self.BAND_PHENOLOGY_1
            elif band in self.phenology:
                band_type = self.BAND_PHENOLOGY_2
            self.feat_type.append(band_type)
            
    
    def generate_features_array(self, tile, **kwargs):
        dtype = kwargs.get("dtype", np.int16)
        fill = kwargs.get("fill", False)
        normalize = kwargs.get("normalize", False)
        standardize = kwargs.get("standardize", False)
        nodata_filter = kwargs.get("nodata_mask", None)
        ds_shape = kwargs.get("shape", None)

        self.fill = fill
        self.standardize = standardize
        self.normalize = normalize

        if ds_shape is not None:
            assert len(ds_shape) == 2, "The provided dataset shape is not valid!"
            self.ds_nrows = ds_shape[0]
            self.ds_ncols = ds_shape[1]

        # Array to hold all features
        features = np.zeros((self.ds_nrows, self.ds_nrows, len(self.feat_indices)), dtype=dtype)

        for feat_index, feat_name, feat_type in zip(self.feat_indices, self.feat_names, self.feat_type):
            feat_nameparts = feat.split(' ')

            if feat_type == self.BAND_PHENOLOGY_1:
                # Phenology 1
                feat = feat_name[self.PHENO_OFFSET:]  # actual feature name
                
                fn = os.path.join(self.pheno_dir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S1.hdf")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                
                # No need to fill missing values, just read the values
                feat_arr = self.read_from_hdf(fn, feat, as_int16=True)  # Use HDF4 method
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)
                
                # Fix values larget than 366
                if feat == 'SOS' or feat == 'EOS' or feat == 'LOS':
                    print(f' --Fixing {feat}.')
                    feat_fixed = self.fix_annual_phenology(feat_arr)
                    feat_arr = feat_fixed[:]

                # Fill missing data
                if self.fill:
                    if feat == 'SOS':
                        print(f'  --Filling {feat}')
                        minimum = 0
                        sos = self.read_from_hdf(fn, 'SOS', as_int16=True)
                        eos = self.read_from_hdf(fn, 'EOS', as_int16=True)
                        los = self.read_from_hdf(fn, 'LOS', as_int16=True)
                        
                        sos_fixed = self.fix_annual_phenology(sos)
                        eos_fixed = self.fix_annual_phenology(eos)
                        los_fixed = self.fix_annual_phenology(los)
                        
                        # # Fix SOS values larger than 365
                        # sos_fixed = np.where(sos > 366, sos-365, sos)

                        # # Fix SOS values larger than 365, needs to be done two times
                        # eos_fixed = np.where(eos > 366, eos-365, eos)
                        # # print(np.min(eos_fixed), np.max(eos_fixed))
                        # if np.max(eos_fixed) > 366:
                        #     eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
                        #     print(f'  --Adjusting EOS again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')

                        filled_sos, filled_eos, filled_los =  self.fill_season(sos_fixed, eos_fixed, los_fixed, minimum,
                                                                        row_pixels=self.tile_rows,
                                                                        max_row=self.tile_rows,
                                                                        max_col=self.tile_cols,
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
                        feat_arr = self.fill_with_mode(feat_arr, 0, row_pixels=self.tile_rows, max_row=self.tile_rows, max_col=self.tile_cols, verbose=False)
                    elif feat == 'GDR' or feat == 'GUR' or feat == 'MAX':
                        # GDR, GUR and MAX should be positive integers!
                        print(f'  --Filling {feat}')
                        feat_arr = self.fill_with_int_mean(feat_arr, 0, var=feat, verbose=False)
                    else:
                        # Other parameters? Not possible
                        print(f'  --Filling {feat}')
                        ds = self.read_from_hdf(fn, feat, as_int16=True)
                        feat_arr = self.fill_with_int_mean(ds, 0, var=feat, verbose=False)

                # Normalize or standardize
                assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                if self.normalize and not self.standardize:
                    feat_arr = self.normalize_dataset(feat_arr)
                elif not self.normalize and self.standardize:
                    feat_arr = self.standardize_dataset(feat_arr)
            
            elif feat_type == self.BAND_PHENOLOGY_2:
                # Phenology 2
                feat = feat_name[self.PHENO_OFFSET:]
                fn = os.path.join(self.phendir, self.phenobased, tile, f"LANDSAT08.PHEN.{self.phenobased}_S2.hdf")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")

                # No need to fill missing values, just read the values
                feat_arr = self.read_from_hdf(fn, feat, as_int16=True)  # Use HDF4 method
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)

                # Fix values larget than 366
                if feat == 'SOS' or feat == 'EOS' or feat == 'LOS':
                    print(f' --Fixing {feat}.')
                    feat_fixed = self.fix_annual_phenology(feat_arr)
                    feat_arr = feat_fixed[:]

                # Extract data and filter by training mask
                if self.fill:
                    # IMPORTANT: Only a few pixels have a second season, thus dataset could
                    # have a huge amount of NaNs, filling will be restricted to replace a
                    # The missing values to NO_DATA
                    print(f'  --Filling {feat}')
                    feat_arr = self.read_from_hdf(fn, feat, as_int16=True)
                    feat_arr = np.where(feat_arr <= 0, self.nodata, feat_arr)
                
                # Normalize or standardize
                assert not (self.normalize and self.standardize), "Cannot normalize and standardize at the same time!"
                if self.normalize and not self.standardize:
                    feat_arr = self.normalize_dataset(feat_arr)
                elif not self.normalize and self.standardize:
                    feat_arr = self.standardize_dataset(feat_arr)
            elif feat_type == self.BAND_SPECTRAL or feat_type == self.BAND_VI:
                # VI or SPECTRAL BAND feature
                feat = feat_name[self.FEAT_OFFSET:]

                if len(feat_nameparts) == 3:
                    month = feat_nameparts[0]
                    band = feat_nameparts[1]
                    fn = os.path.join(self.bands_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')
                elif len(feat_nameparts) == 4:
                    month = feat_nameparts[0]
                    band = feat_nameparts[2][1:-1]  # remove parenthesis
                    fn = os.path.join(self.bands_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')

                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"

                print(f"--{feat_index:>3}: {feat} --> {fn} (as {feat_arr.dtype})")
                
                # Extract data and filter
                feat_arr = self.read_from_hdf(fn, feat, as_int16=True)  # Use HDF4 method
                if nodata_filter is not None:
                    feat_arr = np.where(nodata_filter == 1, feat_arr, self.nodata)

                ### Fill missing data
                if self.fill:
                    minimum = 0  # set minimum for spectral bands
                    if feat_type == self.BAND_VI:
                        minimum = -10000  # minimum for VIs
                    feat_arr = self.fill_with_mean(feat_arr, minimum, var=band.upper(), verbose=False)

                # Normalize or standardize
                if self.normalize:
                    feat_arr = normalize(feat_arr)

            features[:,:,feat_index] = feat_arr
        return features
    

    def read_tiles(self):
        """ Put the extent of tiles into a dictionary 
        Extent format is a dictionary with N, S, W, and E boundaries
        """
        assert os.path.isfile(self.fn_tiles), f"File not found: {self.fn_tiles}"

        self.tiles_extent = {}
        self.tiles = []
        with open(self.fn_tiles, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # print(row)
                row_dict = {}
                for item in row[1:]:
                    itemlst = item.split('=')
                    row_dict[itemlst[0].strip()] = int(float(itemlst[1]))
                self.tiles_extent[row[0]] = row_dict
                self.tiles.append(row[0])
        # print(tiles_extent)


    def read_training_mask(self):
        """ Opens raster sample mask, doesn't change other parameters """
        # WARNING! Assumes spatial reference is same as object's default!
        self.training_mask, _, _, _ = self.open_raster(self.fn_training_mask)
        self.training_mask = self.training_mask.astype(self.dataset.dtype)
    

    def create_tile_dataset(self, **kwargs) -> None:
        save_labels_raster = kwargs.get('save_labels_raster', False)
        save_features = kwargs.get('save_features', False)
        by_season = kwargs.get('by_season', False)
        labels_suffix = kwargs.get('labels_suffix', '') # labels_suffix='data/inegi'
        feat_suffix = kwargs.get('feat_suffix', '')  # feat_suffix='features'

        if by_season:
            # This option in mandatory in this case
            save_features = True
            
        # First, get tiles extent
        self.read_tiles()

        assert self.tiles is not None, "List of tiles is empty (None)."

        # In case sampling was executed in a previous run
        if self.training_mask is None:
            self.read_training_mask()
        # if self.training_labels is None:
        #     self.read_training_labels()

        self.no_data_arr = np.where(self.dataset > self.nodata, 1, self.nodata)  # 1=data, 0=NoData
        # Keep train mask values only in pixels with data, remove NoData
        print(f" Train mask shape: {self.training_mask.shape}")
        self.training_mask = np.where(self.no_data_arr == 1, self.training_mask, self.nodata)

        # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
        print(f"  --no_data_arr={self.no_data_arr.dtype}, training_mask={self.training_mask.dtype} ")
        print(f'  --Training pixels: {(self.training_mask == 1).sum()}')
        print(f'  --No Data pixels: {(self.no_data_arr == 0).sum()}')
        print(f'  --Testing pixels: {(self.training_mask == 0).sum() - (self.no_data_arr == 0).sum()}')

        for tile in self.tiles:
            print(f"\nProcessing tile: {tile}")
            
            # Create new directories
            labels_path = os.path.join(self.output_dir, labels_suffix, tile)
            feat_path = os.path.join(self.output_dir, feat_suffix, self.phenobased, tile)
            if not os.path.exists(labels_path) and save_labels_raster:
                print(f"\nCreating new path: {labels_path}")
                # os.makedirs(labels_path)
            if not os.path.exists(feat_path) and save_features:
                print(f"\nCreating new path: {feat_path}")
                # os.makedirs(feat_path)

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
            tile_landcover = self.dataset[nrow:srow, wcol:ecol]
            print(f"Slice: {nrow}:{srow}, {wcol}:{ecol} {tile_landcover.shape}")
            if save_labels_raster:
                # Save the sliced data into a new raster
                print(f"Writing: {fn_tile}")
                # self.create_raster(fn_tile, tile_landcover, self.spatial_reference, tile_geotransform)
            
            # Slice the training mask and the NoData mask
            tile_training_mask = self.training_mask[nrow:srow, wcol:ecol]
            tile_nodata = self.no_data_arr[nrow:srow, wcol:ecol]

            # Generate features
            tile_features = self.generate_features_array(tile, nodata_mask=tile_nodata, fill=False)

            fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")
            fn_tile_labels = os.path.join(feat_path, f"labels_{tile}.h5")

            # Save the features
            if save_features:
                print(f"Saving {tile} features...")
                # Create (large) HDF5 files to hold all features
                h5_features = h5py.File(fn_tile_features, 'w')
                h5_labels = h5py.File(fn_tile_labels, 'w')

                # Save the training and testing labels
                h5_labels.create_dataset('all', (self.ds_rows, self.ds_cols), data=tile_landcover, dtype=self.dataset.dtype)
                h5_labels.create_dataset('training_mask', (self.ds_rows, self.ds_cols), data=tile_training_mask, dtype=self.dataset.dtype)
                h5_labels.create_dataset('no_data_mask', (self.ds_rows, self.ds_cols), data=tile_nodata, dtype=self.dataset.dtype)

                for n, feature in zip(self.feat_indices, self.feat_names):
                    h5_features.create_dataset(feature, (self.ds_rows, self.ds_cols), data=tile_features[:,:,n])

            if by_season:
                # Aggregate features by season
                # WARNING! Requires HDF5 files to be created first!
                fn_tile_feat_season = os.path.join(feat_path, f"features_season_{tile}.h5")
                self.features_by_season(fn_tile_features, fn_tile_feat_season)


class LandCoverClassifier(FeaturesDataset):
    
    def __init__(self, fn_landcover, bands_dir, pheno_dir, output_dir, **kwargs):
        print("Initializing LandCoverClassifier")
        _features_dir = kwargs.get("features_dir", "features")
        _fn_tiles = kwargs.get("file_tiles", None)
        _phenobased = kwargs.get("phenobased", "NDVI")
        _feat_list = kwargs.get("feat_list", None)
        _fn_training_mask = kwargs.get("training_mask", "")

        self.fn_landcover = fn_landcover

        assert os.path.isdir(bands_dir), f"Directory not found: {bands_dir}"
        assert os.path.isdir(pheno_dir), f"Directory not found: {pheno_dir}"
        assert os.path.isdir(output_dir), f"Directory not found: {output_dir}"
        assert os.path.isfile(fn_landcover), f"File not found: {fn_landcover}"

        FeaturesDataset.__init__(self, self.fn_landcover, bands_dir, pheno_dir, output_dir,
                                 features_dir=_features_dir, file_tiles=_fn_tiles, phenobased=_phenobased,
                                 feat_list=_feat_list, training_mask=_fn_training_mask)
        

    def __str__(self):
        text = ""
        text += FeaturesDataset.__str__
        return text


if __name__ == '__main__':

    # *** Testing code ****
    cwd = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/"
    dir_bands = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/02_STATS/"
    dir_pheno = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/03_PHENO/"
    dir_ancillary = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/data/ancillary/"
    ancillary_dict = {101: ["ag_roi2.tif"], 102: ["roads_roi2.tif", "urban_roi2.tif"]}
    fn_tiles = os.path.join(cwd, 'ROI2/parameters/tiles')

    fn_landcover = os.path.join(cwd, "data/inegi/usv250s7cw2018_ROI2full.tif")

    # # This will incorporate the ancillary data rasters, and carry out the stratified random sampling
    # raster = LandCoverRaster(fn_landcover, cwd)
    # raster.incorporate_ancillary(ancillary_dir=dir_ancillary,
    #                              ancillary_dict=ancillary_dict)
    # raster.sample()
    # print(raster)

    lcc = LandCoverClassifier(fn_landcover, dir_bands, dir_pheno, cwd, file_tiles=fn_tiles)
    print(lcc)
