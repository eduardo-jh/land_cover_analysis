#!/usr/bin/env python
# coding: utf-8

""" Heatmap of the land cover transitions """

import os
import sys
sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import rsmodule as rs

# plt.style.use('ggplot')  # R-like plots

nlabels = [x+0.5 for x in range(0,11)]
print(nlabels)
labels = ['Agriculture',
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

# For the entire Yucatan Peninsula
fn_transitions = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/change/land_cover_class_changes_table.csv"
fn_percent = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/change/land_cover_class_changes_table_percent.csv"
fig_heatmap = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/change/transitions_heatmap.png"
cbr = False
size = (18, 10)

# Inside the CBR
# fn_transitions = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/CBR_land_cover_class_changes_table.csv"
# fn_percent = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/CBR_land_cover_class_changes_table_percent.csv"
# fig_heatmap = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/CBR_transitions_heatmap.png"
# cbr = True

# Outsise the CBR
# fn_transitions = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/SIMAREA_land_cover_class_changes_table.csv"
# fn_percent = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/SIMAREA_land_cover_class_changes_table_percent.csv"
# fig_heatmap = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/SIMAREA_transitions_heatmap.png"
# cbr = True

df_trans = pd.read_csv(fn_transitions, index_col=0)
print(df_trans)
print(df_trans.columns)

df_percent = pd.read_csv(fn_percent, index_col=0)
print(df_percent)
print(df_percent.columns)

dataset = {}
for col in range(len(labels)):
    column = []
    for row in range(len(labels)):
        if df_trans.iloc[row, col] < 0:
            column.append("")
        else:
            column.append(f"{df_trans.iloc[row, col]:,.0f}\n{df_percent.iloc[row, col]:.1f}%")
        # print(f"{row},{col}: {df_trans.iloc[row, col]:,.1f}\n{df_percent.iloc[row, col]:.1f}%")
    dataset[100+col+1] = column
# print(dataset)
df_all = pd.DataFrame(dataset)
print(df_all)

# If CBR remove the last two classes
if cbr:
    df_all = df_all.drop(columns=[110, 111])
    df_all = df_all.drop([9, 10])
    df_percent = df_percent.drop(columns=['110', '111'])
    df_percent = df_percent.drop([110, 111])
    labels.pop()
    labels.pop()
    nlabels.pop()
    nlabels.pop()
    size = (12, 8)

# Plot percentages and annotate with actual values
# fig = plt.figure(figsize=(16, 8), constrained_layout=True)
fig = plt.figure(figsize=size, constrained_layout=True)
# Try: 'PuBuGn', 'BuGn', 'twilight', 'GnBu'. Original: 'crest'
ax = sns.heatmap(df_percent, cmap='PuBuGn', annot=df_all, fmt = '', linewidths=0.5, vmin=0, vmax=100)
# ax = sns.heatmap(df_percent, cmap='crest', annot=df_trans, fmt=",.1f", linewidths=0.5)
# ax.xaxis.tick_top()
ax.set_xticks(nlabels, labels, rotation=90)
ax.set_yticks(nlabels, labels, rotation=0)
ax.set_xlabel("Land cover classes in P1 (2013-2016)")
ax.set_ylabel("Land cover classes in P3 (2019-2022)")


plt.savefig(fig_heatmap, bbox_inches='tight', dpi=100)
# plt.savefig(fig_heatmap, bbox_inches='tight', dpi=150)
plt.close()
