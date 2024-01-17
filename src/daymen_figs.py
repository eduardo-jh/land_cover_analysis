#!/usr/bin/env python
# coding: utf-8

# Create figures with tmin, tmax, and precipitation averages for the Yucatan Peninsula

import sys
import os

sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib')

import rsmodule as rs

# Directories
cwd = "D:/Downloads/Daymet"
vars = ["prcp", "tmax", "tmin"]

# List the files in each directory
for var in vars:
    subdir = os.path.join(cwd, var)
    only_files = rs.get_files(subdir, "tif")
    
    # print(f"\nDIR: {subdir}\nFiles:")
    # for f in only_files:
    #     print(f"{f}")
    
    # Open the files
    