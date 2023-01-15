import sys
# adding the directory with modules
sys.path.insert(0, '/vipdata/2023/lc_classif/lib/')
from rsmodule import open_raster, create_raster

fn_landcover = '/vipdata/2023/CALAKMUL/ROI1/TRAIN/usv250s7cw_ROI1_LC_KEY.tif'

train_mask, nodata, metadata, geotransform, projection = open_raster(fn_landcover)
print(f'Opening raster: {fn_landcover}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {train_mask.shape[1]}')
print(f'Rows          : {train_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')

# Get percentage of each land cover class in the raster

quadrants = 9