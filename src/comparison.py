#!/usr/bin/env python
# coding: utf-8

""" Raster comparison

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2023-10-23
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
import rsmodule as rs

def difference(fn_raster1, fn_raster2, fn_diff, **kwargs):
    _title = kwargs.get("title", "")
    ds1, nodata, geotransform, spatial_reference = rs.open_raster(fn_raster1)
    ds2, _, _, _ = rs.open_raster(fn_raster2)
    print(ds1.dtype, ds2.dtype)
    ds1 = ds1.astype(np.int16)
    ds2 = ds2.astype(np.int16)

    assert ds1.shape == ds2.shape, "Shape doesn't match!"
    diff = ds2 - ds1
    diff = diff.filled(0)

    print(np.unique(diff, return_counts=True))

    print(f"Difference raster: {fn_diff}")
    rs.create_raster(fn_diff, diff, spatial_reference, geotransform)

    fn_plot = fn_diff[:-4] + ".png"
    print(f"Difference plot: {fn_plot}")
    rs.plot_dataset(diff, savefig=fn_plot, title=_title)


def get_spectral_signature(x, y):
   pass


if __name__ == "__main__":
  cwd = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/"
#   fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"
#   fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11_ancillary.tif"
#   fn_landcover3 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11_ancillary.tif"

  fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_allclasses.tif"
  fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_allclasses.tif"
  fn_landcover3 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_allclasses.tif"

  dir_comp = os.path.join(cwd, "comparison")
  if not os.path.exists(dir_comp):
    print(f"\nCreating new path: {dir_comp}")
    os.makedirs(dir_comp)
  
  diff12 = os.path.join(dir_comp, "diff12.tif")
  difference(fn_landcover2, fn_landcover1, diff12, title="Difference between periods 2013-2016 and 2016-2019")

  diff23 = os.path.join(dir_comp, "diff23.tif")
  difference(fn_landcover3, fn_landcover2, diff23, title="Difference between periods 2016-2019 and 2019-2022")

  diff13 = os.path.join(dir_comp, "diff13.tif")
  difference(fn_landcover3, fn_landcover1, diff13, title="Difference between periods 2013-2016 and 2019-2022")