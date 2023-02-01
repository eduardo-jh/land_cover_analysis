#!/usr/bin/env python
# coding: utf-8

# As Jan 31, 2023: All the functions of this file are incorporated to rsmodule.py

import csv
import numpy as np
import matplotlib.pyplot as plt
from rsmodule import open_raster, create_raster

def plot_hbar(x, y, fname, title_='', xlabel_='', xlim_=None):
    # Create a horizontal bar plot
    plt.figure(figsize=(12, 24), constrained_layout=True)
    pl = plt.barh(x, y)
    for bar in pl:
        value = bar.get_width()
        text = round(value, 4)
        if value > 0.01:
            text = round(value, 2)
        plt.annotate(text, xy=(value+0.1, bar.get_y()+0.25))
    plt.title(title_)
    plt.xlabel(xlabel_)
    if xlim_ is not None:
        plt.xlim(xlim_)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close()

cwd = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/'
print(f'Reading GAP/LANDFIRE attribute table...', end='')
fn = cwd + 'ML/gaplf2011lc_v30_lcc_15/GAP_LANDFIRE_National_Terrestrial_Ecosystems_2011_Attributes.txt'

# Create a dictionary with values and their land cover ecosystem names
land_cover_classes = {}
with open(fn, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    header = next(reader)
    # print(f'Header: {",".join(header)}')
    for row in reader:
        key = int(row[0])  # Keys are ecosystem, a numeric value
        val = row[18]  # Values are the descriptions, a string
        grp = row[16]  # Groups, also string
        # print(f'{key}: {val}')
        # land_cover_classes[key] = val
        land_cover_classes[key] = [val, grp]
print(f'{len(list(land_cover_classes.keys()))} items read.')

# Open the land cover raster and retrive the land cover classes
raster = cwd + 'ML/ROI_gaplf2011lc_v30_lcc_15.tif'  # File resampled and clipped externally
raster_arr, nodata, metadata, geotransform, projection = open_raster(raster)
print(f'Opening raster: {raster}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {raster_arr.shape[1]}')
print(f'Rows          : {raster_arr.shape[0]}')
print(f'Geotransform  : {geotransform}')
print(f'Projection    : {projection}')

# First get the land cover keys in the array, then get their corresponding description
lc_keys_arr, frequency = np.unique(raster_arr, return_counts=True)
print(f'{len(lc_keys_arr)} unique land cover values in ROI.')

land_cover = {}
land_cover_groups = {}
for lc_key, freq in zip(lc_keys_arr, frequency):
    # Retrieve land cover ecosystem and its group
    ecosystem = land_cover_classes[lc_key][0]
    group = land_cover_classes[lc_key][1]
    
    print(f'KEY={lc_key:>3} [FREQ={freq:>10}]: {ecosystem:>75} GROUP={group:<75} ', end='')
    land_cover[freq] = [lc_key, ecosystem, group]

    if land_cover_groups.get(group) is None:
        print(f'NEW group.')
        land_cover_groups[group] = freq
    else:
        land_cover_groups[group] += freq
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
print('Saving statistics file...')
stats = cwd + 'ML/san_juan_gap_lc_dist.csv'
with open(stats, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Key', 'Description', 'Group', 'Frequency', 'Percentage'])
    for i in range(len(counts)):
        # print(f'{lc_description[i]}: {lc_frequency[i]}, {percentages[i]}')
        # Write each line with the land cover key, its description, group, pixel frequency, and percentage cover
        writer.writerow([lc_keys[i], lc_description[i], lc_group[i], lc_frequency[i], percentages[i]])

# Create a bar plot using the values from raster in array form
print('Plotting...')
plot_hbar(lc_description, percentages, cwd + 'ML/san_juan_gap_lc_dist.png', 'GAP/LANDFIRE Land Cover Classes in San Juan River', 'Percentage (based on pixel count)', (0,30))


# Now flip the groups dictionary to use frequency as key, and group as value
key_grps = list(land_cover_groups.keys())
lc_grps_by_freq = {}
for grp in key_grps:
    lc_grps_by_freq[land_cover_groups[grp]] = grp

# Create lists
grp_filter = []
frq_lc = []
threshold = 1000 # Remove classes with few pixels
print(f'Removing classes with pixel count less than {threshold}')
grp_key_freq = sorted(list(lc_grps_by_freq.keys()))
for freq in grp_key_freq:
    if freq >= threshold:
        grp_filter.append(lc_grps_by_freq[freq])
        frq_lc.append(freq)
    else:
        print(f'Group "{lc_grps_by_freq[freq]}" removed by small pixel count {freq}')
print(f'{len(grp_filter)} land cover groups added.')

# Calculate percentage based on pixel count of each land cover group
percent_grp = (frq_lc / sum(frq_lc)) * 100.

print(f'Plotting groups...')
plot_hbar(grp_filter, percent_grp, cwd + 'ML/san_juan_gap_lc_groups.png', 'GAP/LANDFIRE Land Cover Classes (by group) in San Juan River')

# Reclassify rasters to use land cover groups
print('Creating reclassification key...')
ecos_by_group = {}
with open(stats, 'r') as csvfile:
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

fn_grps = f'/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/GAP/gaplf2011lc_v30_groups.tif'
raster_groups = np.zeros(raster_arr.shape, dtype=np.int64)
src_proj = 32612  # WGS 84 / UTM zone 12N

print('Saving the group keys...')
stats = cwd + f'ML/group_keys_{len(ecos_by_group.keys())}_groups.csv'
with open(stats, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Group Key', 'Description', 'Ecosystems'])

    for i, grp in enumerate(sorted(list(ecos_by_group.keys()))):
        group = i+1
        print(f'{group:>3} {grp:>75} {ecos_by_group[grp]}')
        raster_to_replace = np.zeros(raster_arr.shape, dtype=np.int64)

        writer.writerow([group, grp, ','.join(str(x) for x in ecos_by_group[grp])])
        
        # Join all the ecosystems of the same group
        for ecosystem in ecos_by_group[grp]:
            raster_to_replace[np.equal(raster_arr, ecosystem)] = group
            raster_groups[np.equal(raster_arr, ecosystem)] = group
            print(f'Replacing {ecosystem} with {group}')

        # WARNING! THIS BLOCK WILL CREATE A RASTER FILE PER LAND COVER CLASS
        group_str = str(i+1).zfill(3)
        fn = f'/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/GAP/gaplf2011lc_v30_grp_{group_str}.tif'
        print(f'Creating raster for group {group} in {fn} ...')
        create_raster(fn, raster_to_replace, src_proj, geotransform)
print(f'Creating raster for groups {fn_grps} ...')
create_raster(fn_grps, raster_groups, src_proj, geotransform)
print('Enjoy ;-)')
# DONE
