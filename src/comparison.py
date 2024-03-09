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
    """ Computes the difference between two rasters by substracting raster 2 from 1 """
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



def compare(fn_raster1, fn_raster2, outdir, label):
   """ Compares two raster files (two land cover maps) """

   # Make a Chi-Square test for homogeneity, see if they are different
   print(f"=== CHANGE BETWEEN: {fn_raster1} AND {fn_raster2}")
   rs.chi_square_raster(fn_raster1, fn_raster2, outdir, label)
   

if __name__ == "__main__":
  cwd = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/"
#   fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"
#   fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11_ancillary.tif"
#   fn_landcover3 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11_ancillary.tif"

  # fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_allclasses.tif"
  # fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_allclasses.tif"
  # fn_landcover3 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_allclasses.tif"

  # dir_comp = os.path.join(cwd, "comparison")
  # if not os.path.exists(dir_comp):
  #   print(f"\nCreating new path: {dir_comp}")
  #   os.makedirs(dir_comp)
  
  # diff12 = os.path.join(dir_comp, "diff12.tif")
  # difference(fn_landcover2, fn_landcover1, diff12, title="Difference between periods 2013-2016 and 2016-2019")

  # diff23 = os.path.join(dir_comp, "diff23.tif")
  # difference(fn_landcover3, fn_landcover2, diff23, title="Difference between periods 2016-2019 and 2019-2022")

  # diff13 = os.path.join(dir_comp, "diff13.tif")
  # difference(fn_landcover3, fn_landcover1, diff13, title="Difference between periods 2013-2016 and 2019-2022")

  
  # =============================== COMPARISON 2016-2019 vs MAD-MEX ===============================
  # comp_id = "Preds_vs_MADMEX"
  # output_dir = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/comparison/"
  # fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/comparison/maxmex_landsat_2018_ROI2_reclass.tif"
  # fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/results/2023_10_28-18_19_05/2023_10_28-18_19_05_predictions.tif"

  # # =============================== COMPARISON 2019-2022 vs NALCMS ===============================
  comp_id = "Preds_vs_NALCMS"
  output_dir = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/comparison/"
  fn_landcover1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/comparison/yucatan_peninsula_nalcms2020_reclass2.tif"
  fn_landcover2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/results/2023_10_29-12_10_07/2023_10_29-12_10_07_predictions.tif"

  # =============================== RUN COMPARISON ===============================
  compare(fn_landcover1, fn_landcover2, output_dir, comp_id)

  # Use a similar approach to validation to compare against the validation raster.
  # Validation rasters are genereated from point vectors into raster of radii of 30m (single-pixel),
  # 45m (3x3 window), 105m (7x7 window), in order to try simulate patches/windows of pixels
  # Smaller windows may ignore some points in the process of generating

  # ======================= PREPARE VALIDATION FOR 2016-2019 vs MAD-MEX =======================
  # validation_path = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/validation/"
  
  # fn_valid_map = os.path.join(validation_path, 'infys_pts_2015_2016.tif')
  # id="MADMEX_1x1"

  # fn_valid_map = os.path.join(validation_path, 'infys_pts_2015_2016_r45_3x3.tif')
  # id="MADMEX_3x3"

  # fn_valid_map = os.path.join(validation_path, 'infys_pts_2015_2016_r75_5x5.tif')
  # id="MADMEX_5x5"

  # ======================= PREPARE VALIDATION FOR 2019-2022 vs NALCMS =======================
  validation_path = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/validation/"
  
  # fn_valid_map = os.path.join(validation_path, 'infys_pts_2017_2019.tif')
  # id="NALCMS_1x1"

  # fn_valid_map = os.path.join(validation_path, 'infys_pts_2017_2019_r45_3x3.tif')
  # id="NALCMS_3x3"

  fn_valid_map = os.path.join(validation_path, 'infys_pts_2017_2019_r75_5x5.tif')
  id="NALCMS_5x5"

  # =============================== RUN VALIDATION ===============================
  rs.validation_raster(output_dir,fn_landcover1, fn_valid_map, prefix=id)
  print("All done! ;-)")