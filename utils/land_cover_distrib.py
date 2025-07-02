#!/usr/bin/env python
# coding: utf-8

""" Generates land cover plot using sorted values """

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
import rsmodule as rs

plt.rcParams.update({'font.size': 16})
plt.style.use('ggplot')  # R-like plots

cwd = "/VIP/engr-didan02s/DATA/EDUARDO/2024/YUCATAN_LAND_COVER/ROI2/"

print("Openning rasters...")
r1 = os.path.join(cwd, "2013_2016", "results", "2024_03_06-23_19_43", "2024_03_06-23_19_43_predictions.tif")
r2 = os.path.join(cwd, "2016_2019", "results", "2024_03_08-13_29_31", "2024_03_08-13_29_31_predictions.tif")
r3 = os.path.join(cwd, "2019_2022", "results", "2024_03_12-19_32_01", "2024_03_12-19_32_01_predictions.tif")

ds1, _, _, _ = rs.open_raster(r1)
ds2, _, _, _ = rs.open_raster(r2)
ds3, _, _, _ = rs.open_raster(r3)

print("Counting class frequencies...")
vals1, counts1 = np.unique(ds1, return_counts=True)
vals2, counts2 = np.unique(ds2, return_counts=True)
vals3, counts3 = np.unique(ds3, return_counts=True)

print("Frequecies 1:", vals1, counts1)
print("Frequecies 2:", vals2, counts2)
print("Frequecies 3:", vals3, counts3)

df1 = pd.DataFrame({"Classes": vals1, "P1 (2013-2016)": counts1})
df2 = pd.DataFrame({"Classes": vals2, "P2 (2016-2019)": counts2})
df3 = pd.DataFrame({"Classes": vals3, "P3 (2019-2022)": counts3})

df4 = pd.merge(df1, df2, on="Classes")
df = pd.merge(df4, df3, on="Classes")
# Remove the last row
df.drop(11, axis=0, inplace=True)
# df["Classes"] = df["Classes"].astype(int)
df["Classes"] = ['Agriculture',
          'Urban',
          'Water',
          'Barren land',
          'Mangrove',
          'Evergreen tropical forest',
          'Savanna',
          'Wetland',
          'Deciduous tropical forest',
          'Coastal vegetation',
          'Oak forest']
df.set_index("Classes", drop=True, inplace=True)
df.sort_values("P1 (2013-2016)", inplace=True)
df.to_csv(os.path.join(cwd, "LandCover_Distribution_Predictions_Values.csv"))

# # Calculate percentages
# df["P1 (2013-2016)"] = df["P1 (2013-2016)"] / df["P1 (2013-2016)"].sum() * 100
# df["P2 (2016-2019)"] = df["P2 (2016-2019)"] / df["P2 (2016-2019)"].sum() * 100
# df["P3 (2019-2022)"] = df["P3 (2019-2022)"] / df["P3 (2019-2022)"].sum() * 100
# print(df)
# df.to_csv(os.path.join(cwd, "LandCover_Distribution_Predictions.csv"))

# p = df.plot.barh()

# plt.xlabel("Percent")
# plt.ylabel("")
# # plt.gca().set_xscale('log')
# plt.legend(loc="lower right")
# plt.savefig(os.path.join(cwd, "LandCover_Distribution_Predictions.png"), bbox_inches='tight', dpi=150)
