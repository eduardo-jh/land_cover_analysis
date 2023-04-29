#!/usr/bin/env python
# coding: utf-8
""" Code to test the fill NaNs functions in HDF5 files """

# import sys
# import platform
# import h5py
import pandas as pd
import numpy as np
# from scipy import stats

import matplotlib.pyplot as plt

fn = '/vipdata/2023/CALAKMUL/ROI1/results/2023_04_28-20_46_30_rf_confussion_table.csv'

data = pd.read_csv(fn, header=None)

values = np.array(data)

normalized = (values - np.min(values)) / (np.max(values) - np.min(values))

# print(normalized.shape)
# print(data.info())
# print(data.head())

land_cover = [x for x in range(0, 26)]

fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(normalized)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(land_cover)), labels=land_cover)
ax.set_yticks(np.arange(len(land_cover)), labels=land_cover)
ax.grid(False)

# Loop over data dimensions and create text annotations.
for i in range(len(land_cover)):
    for j in range(len(land_cover)):
        text = ax.text(j, i, f'{normalized[i, j]:0.1f}',
                       ha="center", va="center", color="w")
        
# plt.colorbar()
# plt.savefig(save_preds_fig, bbox_inches='tight', dpi=300)
ax.set_title("Cronfusion table")
plt.show()
plt.close()