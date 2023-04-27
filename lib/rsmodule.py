#!/usr/bin/env python
# coding: utf-8

""" NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.4.1 & Matplotlib 3.5.2 from conda-forge)

Some remote sensing and GIS utilities

Eduardo Jimenez <eduardojh@email.arizona.edu>
Changelog:
    Jul 12, 2022: Quality assessment of 'pixel_qa', 'sr_aerosol' and 'radsat_qa' bands
    Aug 15, 2022: Creation of MODIS-like QA using Landsat's 'pixel_qa' and 'sr_aerosol'
    Jan 13, 2023: Functions to prepare raster datset for training machine learning (functions from 'San Juan River' script)
    Jan 17, 2023: Land cover percentage analysis on training rasters and new format in function definitions
"""
import gc
import os
import platform
system = platform.system()
if system == 'Windows':
    # On Windows 10
    os.environ['PROJ_LIB'] = 'D:/anaconda3/envs/rsml/Library/share/proj'
    os.environ['GDAL_DATA'] = 'D:/anaconda3/envs/rsml/Library/share'
elif system == 'Linux':
    # On Ubuntu Workstation
    os.environ['PROJ_LIB'] = '/home/eduardo/anaconda3/envs/rsml/share/proj/'
    os.environ['GDAL_DATA'] = '/home/eduardo/anaconda3/envs/rsml/share/gdal/'
# elif system == 'Linux':
#     # On Alma Linux Server
#     os.environ['PROJ_LIB'] = '/home/eduardojh/.conda/envs/rsml/share/proj/'
#     os.environ['GDAL_DATA'] = '/home/eduardojh/.conda/envs/rsml/share/gdal/'
else:
    print('System not yet configured!')
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pyhdf.SD import SD, SDC
from matplotlib.colors import ListedColormap
from osgeo import gdal
from osgeo import osr

plt.style.use('ggplot')

def open_raster(filename: str) -> tuple:
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
    
    # Mask 'NoData' values
    raster = np.ma.masked_values(raster_array, nodata)

    # Clean
    del(dataset)
    del(raster_array)
    gc.collect()

    return raster, nodata, metadata, geotransform, projection


def show_raster(filename: str, **kwargs) -> None:
    
    _savefigs = kwargs.get('savefigs', '')
    _cmap = kwargs.get('savefigs', 'viridis')
    _dpi = kwargs.get('dpi', 300)
    _size = kwargs.get('figsize', (12,12))
    _verbose = kwargs.get('verbose', False)

    print(f'\n *** Openning {filename} ***')
    
    # Open the raster, read it as array and get the geotransform
    raster_arr, nd, meta, gt = open_raster(filename)

    # Get the raster extent
    rows, cols = raster_arr.shape
    ulx, xres, _, uly, _, yres = gt
    extent = [ulx, ulx + xres*cols, uly, uly + yres*rows]

    if _verbose:
        print(f'Metadata: {meta}')
        print(f'NoData  : {nd}')
        print(f'Columns : {cols}')
        print(f'Rows    : {rows}')
        print(f'Extent  : {extent}')
    
    # Display with matplotlib
    if _savefigs != '':
        plt.figure(figsize=_size)
        plt.imshow(raster_arr, cmap=_cmap)
        plt.colorbar()
        # plt.show()
        plt.savefig(_savefigs, bbox_inches='tight', dpi=_dpi)
        plt.close()


def create_raster(filename: str, data: np.ndarray, epsg: int, geotransform: list, **kwargs) -> None:
    """ Create a raster (GeoTIFF) from a numpy array """
    _verbose = kwargs.get('verbose', False)

    driver_gtiff = gdal.GetDriverByName('GTiff')

    rows, cols = data.shape

    ds_create = driver_gtiff.Create(filename, xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Byte)

    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds_create.SetProjection(srs.ExportToWkt())

    # Set the geotransform
    ds_create.SetGeoTransform(geotransform)

    if _verbose:
        print(f'Type of driver: {type(driver_gtiff)}')
        print(ds_create.GetProjection())
        print(ds_create.GetGeoTransform())

    ds_create.GetRasterBand(1).WriteArray(data)  # write the array to the raster
    ds_create.GetRasterBand(1).SetNoDataValue(0)  # set the no data value
    ds_create = None  # properly close the raster

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
    print(f'Unique values in raster: {unique_values}')
    
    # Create an array to save selected pixels
    pixel_qa_mask = np.zeros(raster.shape, dtype=np.int16)
    # Apply same mask as 'raster' for 'NoData'
    pixel_qa_mask = np.ma.masked_array(pixel_qa_mask, mask=np.ma.getmask(raster))

    # Apply each filter
    for i, filter in enumerate(filters):
        
        # Apply only filters in the raster's unique values
        if not filter in unique_values:
            if _verbose:
                print(f'Filter {filter} ({i+1} of {len(filters)}) not found. Skipping...')
            continue

        # filtered_qa = raster & filter # Will include similar binary values, avoid
        filtered_qa = np.equal(raster, filter)  # Get the exact value

        if _verbose:
            print(f'Filter: {filter} ({i+1} of {len(filters)})')
            # Get unique values
            filtered_arr = np.ma.getdata(filtered_qa)
            uniques = np.unique(filtered_arr)
            # uniques = np.unique(filtered_qa)
            print(f'{len(uniques)} unique value(s): {uniques}')

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
    print(f'{title}  max: {array.max():.2f}, min:{array.min():.2f} avg: {array.mean():.4f} std: {array.std():.4f}')

def read_from_hdf(filename: str, var) -> np.ndarray:
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
        print('Generating intermediate plots')
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
        print('Generating rank plots')
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

def land_cover_percentages(raster_fn: str, keys_fn: str, stats_fn: str, **kwargs) -> tuple:   
    """ Calculate the land cover percentages from a raster_fn file

    :param str raster_fn: name of the raster file (GeoTIFF) with the land cover classes
    :param str keys_fn: name of a tab delimited text file that links raster keys (numeric) and land cover classes
    :param str stats_fn: name to save a file with statistics (CSV)
    :param str plot: name to save a bar plot with land cover percentages (PNG)
    :param tuple indices: column numbers for land cover column, land cover key column, and group column (GAP or INEGI)
    """
    _indices = kwargs.get('indices', (0,18,16))  # default values are for GAP/LANDCOVER attibutes text file
    _verbose = kwargs.get('verbose', False)

    print(f'\nCalculating land cover percentages...')

    # Unzip the column indexes from tuple
    col_key, col_val, col_grp = _indices

    # Create a dictionary with values and their land cover ecosystem names
    land_cover_classes = {}
    with open(keys_fn, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)
        if _verbose:
            print(f'  Header: {",".join(header)}')
        for row in reader:
            # Skip blank lines
            if len(row) == 0:
                continue

            key = int(row[col_key])
            val = row[col_val]
            grp = row[col_grp]

            # Too much to show
            # if _verbose:
            #     print(f'  {key}: {val}')
            
            # land_cover_classes[key] = val
            land_cover_classes[key] = [val, grp]
    unique_classes = list(land_cover_classes.keys())
    
    if _verbose:
        print(f'  Done. {len(unique_classes)} unique land cover classses read.')

    # Open the land cover raster and retrive the land cover classes
    raster_arr, nodata, metadata, geotransform, projection = open_raster(raster_fn)
    if _verbose:
        print(f'  Opening raster: {raster_fn}')
        print(f'  Metadata      : {metadata}')
        print(f'  NoData        : {nodata}')
        print(f'  Columns       : {raster_arr.shape[1]}')
        print(f'  Rows          : {raster_arr.shape[0]}')
        print(f'  Geotransform  : {geotransform}')
        print(f'  Projection    : {projection}')

    # First get the land cover keys in the array, then get their corresponding description
    lc_keys_arr, frequency = np.unique(raster_arr, return_counts=True)

    if _verbose:
        print(f'  {lc_keys_arr}')
        print(f'  {len(lc_keys_arr)} unique land cover values in ROI.')

    land_cover = {}
    land_cover_groups = {}
    for lc_key, freq in zip(lc_keys_arr, frequency):
        # Skip the MaskedConstant objects
        if not lc_key in unique_classes:
            if _verbose:
                print(f'  Skip the MaskedConstant object: {lc_key}')
            continue
        # Retrieve land cover ecosystem and its group
        ecosystem = land_cover_classes[lc_key][0]
        group = land_cover_classes[lc_key][1]
        
        if _verbose:
            print(f'  KEY={lc_key:>3} [FREQ={freq:>10}]: {ecosystem:>75} GROUP={group:<75} ', end='')
        land_cover[freq] = [lc_key, ecosystem, group]

        if land_cover_groups.get(group) is None:
            if _verbose:
                print(f'NEW group.')
            land_cover_groups[group] = freq
        else:
            land_cover_groups[group] += freq
            if _verbose:
                print(f'EXISTING group.')

    # Calculate percentage based on pixel count of each land cover
    counts = sorted(list(land_cover.keys()))
    total = sum(counts)
    percentages = (counts / total) * 100.

    # Create lists of land cover key, its description, group, and pixel frequency
    lc_keys = []
    lc_description = []
    lc_group = []
    lc_frequency = []
    for key_counts in counts:
        lc_keys.append(land_cover[key_counts][0])
        lc_description.append(land_cover[key_counts][1])
        lc_group.append(land_cover[key_counts][2])
        lc_frequency.append(key_counts)
    # print(lc_group)
    # print(lc_description)

    # Save a file with statistics
    print('  Saving land cover statistics file...')
    # WARNING! Windows needs "newline=''" or it will write \r\r\n which writes an empty line between rows
    with open(stats_fn, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Key', 'Description', 'Group', 'Frequency', 'Percentage'])
        for i in range(len(counts)):
            # print(f'{lc_description[i]}: {lc_frequency[i]}, {percentages[i]}')
            # Write each line with the land cover key, its description, group, pixel frequency, and percentage cover
            writer.writerow([int(lc_keys[i]), lc_description[i], lc_group[i], lc_frequency[i], percentages[i]])
    print(f'Calculating land cover percentages... done!')

    return lc_description, percentages, land_cover_groups, raster_arr, geotransform

def land_cover_percentages_grp(land_cover_groups: dict, threshold: int = 1000, **kwargs) -> tuple:
    """ Calculate land cover percentages by group
    :param dict land_cover_groups:
    :param int threshold: minimum pixel count for a land cover class to be considered in a group
    """
    _verbose = kwargs.get('verbose', False)

    print('\nCalculating land cover percentages per group...')

    # Now flip the groups dictionary to use frequency as key, and group as value
    key_grps = list(land_cover_groups.keys())
    lc_grps_by_freq = {}
    for grp in key_grps:
        lc_grps_by_freq[land_cover_groups[grp]] = grp

    # Create lists
    grp_filter = []
    frq_lc = []
    if _verbose:
        print(f'  Removing classes with pixel count less than {threshold}')
    grp_key_freq = sorted(list(lc_grps_by_freq.keys()))
    for freq in grp_key_freq:
        if freq >= threshold:
            grp_filter.append(lc_grps_by_freq[freq])
            frq_lc.append(freq)
        else:
            if _verbose:
                print(f'  Group "{lc_grps_by_freq[freq]}" removed by small pixel count: {freq}')
    
    if _verbose:
        print(f'  {len(grp_filter)} land cover groups added.')

    # Calculate percentage based on pixel count of each land cover group
    percent_grp = (frq_lc / sum(frq_lc)) * 100.
    print('Calculating land cover percentages per group... done!')

    return grp_filter, percent_grp

def reclassify_land_cover_by_group(raster_arr: np.ndarray, raster_geotransform: list, raster_proj: int, grp_filter: list, fn_lc_stats: str, fn_grp_keys: str, fn_grp_landcover: str, **kwargs) -> None:
    """ Creates a reclassified land cover raster using groups (groups of land cover)

    :param str intermediate: base name for intermediate rasters, will create one per class if non empty
    """
    _intermediate = kwargs.get('intermediate', '')
    _verbose = kwargs.get('verbose', False)

    print('\nReclassifying rasters to use land cover groups...')
    
    # Creating reclassification key
    ecos_by_group = {}
    with open(fn_lc_stats, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            key = int(row[0])  # Keys are ecosystem, a numeric value
            grp = row[2]  # Groups, also string

            # Use the groups filter created before, in order to
            # discard the groups with lower pixel count
            if not grp in grp_filter:
                continue
            
            if ecos_by_group.get(grp) is None:
                # Create new group
                ecos_by_group[grp] = [key]
            else:
                # Add the ecosystem key to the group
                ecos_by_group[grp].append(key)

    raster_groups = np.zeros(raster_arr.shape, dtype=np.int64)

    print('  Saving the group keys...')
    # WARNING! Windows needs "newline=''" or it will write \r\r\n which writes an empty line between rows
    with open(fn_grp_keys, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Group Key', 'Description', 'Ecosystems'])

        for i, grp in enumerate(sorted(list(ecos_by_group.keys()))):
            group = i+1
            if _verbose:
                print(f'  Group key: {group:>3}, Description: {grp}, Classes|Ecosystems: {ecos_by_group[grp]}')
            raster_to_replace = np.zeros(raster_arr.shape, dtype=np.int64)

            writer.writerow([group, grp, ','.join(str(x) for x in ecos_by_group[grp])])
            
            # Join all the ecosystems of the same group
            for ecosystem in ecos_by_group[grp]:
                raster_to_replace[np.equal(raster_arr, ecosystem)] = group
                raster_groups[np.equal(raster_arr, ecosystem)] = group
                if _verbose:
                    print(f'  --Replacing {ecosystem} with {group}')

            # WARNING! THIS BLOCK WILL CREATE A RASTER FILE PER LAND COVER CLASS
            if _intermediate != '':
                # If base name given, create intermediate rasters
                group_str = str(i+1).zfill(3)
                fn_interm_raster = f'{_intermediate}_{group_str}.tif'
                if _verbose:
                    print(f'  Creating raster for group {group} in {fn_interm_raster} ...')
                create_raster(fn_interm_raster, raster_to_replace, raster_proj, raster_geotransform)

    if _verbose:
        print(f'  Creating raster for groups {fn_grp_landcover} ...')
    create_raster(fn_grp_landcover, raster_groups, raster_proj, raster_geotransform)
    print('Reclassifying rasters to use land cover groups... done!')


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


if __name__ == '__main__':
    # --*-- TESTING CODE --*--

    ############################### Create MODIS-like QA from Landsat ###################################
    print('Creating MODIS-like QA raking from LANDSAT')
    directory = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/021046'

    fn1 = 'LC08_L1TP_021046_20130415_20170310_01_T1_pixel_qa.tif'
    fn_raster_qa = os.path.join(directory, fn1)
    pixel_arr, pixel_nd, pixel_meta, pixel_gt = open_raster(fn_raster_qa)

    fn2 = 'LC08_L1TP_021046_20130415_20170310_01_T1_sr_aerosol.tif'
    fn_raster_aerosol = os.path.join(directory, fn2)
    aerosol_arr, aerosol_nd, aerosol_meta, aerosol_gt = open_raster(fn_raster_aerosol)

    print('Creating ranks...')
    rank_qa = create_qa(pixel_arr, aerosol_arr)
    print(f'Ranks: {np.unique(rank_qa)}')

    # Create a personalized colormap for the QA Rank
    col_dict={0:"green", 1:"blue", 2:"yellow", 3:"grey"}
    lbls = np.array(["0", "1", "2", "3"])
    save_cmap_array(rank_qa, fn1[:-13], col_dict, lbls, title='QA Rank', suffix='_rank')

