#!/usr/bin/env python
# coding: utf-8

""" Plot for the percentages of transition in the Calakmul Biosphere Reserve """

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn")
# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic',
# 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright',
# 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
# 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
# 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

cwd = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/"

# ========== Create plots for Evergreen Tropical Forest (Hectares) ==========

# fn_CBR = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table.csv")
# fn_SIMAREA = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table.csv")
# fn_CBR_percent = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table_percent.csv")
# fn_SIMAREA_percent = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table_percent.csv")
# fn_save_bars = os.path.join(cwd, "effectiveness", "effectiveness_comparison_evergreen.png")

# # Read the percentages
# df_CBR = pd.read_csv(fn_CBR, index_col=0)
# df_SIMAREA = pd.read_csv(fn_SIMAREA, index_col=0)
# df_CBR_percent = pd.read_csv(fn_CBR_percent, index_col=0)
# df_SIMAREA_percent = pd.read_csv(fn_SIMAREA_percent, index_col=0)
# print(df_CBR)
# print(df_SIMAREA.info())

# class_names = list(df_CBR.columns)
# num_classes = np.array([int(x) for x in class_names])
# print(f"Classes: {num_classes}")

# # Create a new dataframe for comparison
# df_comparison = pd.DataFrame(df_CBR['106'])
# df_comparison.columns = ['CBR']
# df_comparison['SIMAREA'] = df_SIMAREA['106']
# df_comparison['CBR_percent'] = df_CBR_percent['106']
# df_comparison['SIMAREA_percent'] = df_SIMAREA_percent['106']
# print(df_comparison)

# # Select a subset of classes
# class_names = ['101', '102', '109']
# long_class_names = ['Agriculture', 'Urban', 'Deciduous tropical forest']
# num_classes = np.array([int(x) for x in class_names])
# df_ETF = df_comparison.loc[num_classes]
# # Reset index values and get them for plotting
# df_ETF.reset_index(inplace=True)
# num_classes = df_ETF.index.values
# print(df_ETF)

# # Labels
# r1_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_ETF['CBR'], df_ETF['CBR_percent'])]
# r2_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_ETF['SIMAREA'], df_ETF['SIMAREA_percent'])]
# print(r1_labels)
# print(r2_labels)

# #  Create a barplot
# width = 0.4
# fig, ax = plt.subplots(layout='constrained')
# r1 = ax.bar(num_classes-width/2, df_ETF['CBR'], width, label="Inside")#, log=True)
# r2 = ax.bar(num_classes+width/2, df_ETF['SIMAREA'], width, label="Outside")#, log=True)
# ax.bar_label(r1, r1_labels, padding=3, fontsize=10)
# ax.bar_label(r2, r2_labels, padding=3, fontsize=10)
# ax.set_ylabel("Area of Change (ha)")
# ax.set_xticks(num_classes, long_class_names)
# ax.set_title("Loss from Evergreen Tropical Forest")
# ax.legend(loc='upper left', ncol=2)
# plt.savefig(fn_save_bars, bbox_inches='tight', dpi=150)

# ========== Create plots for Deciduous Tropical Forest (Hectares) ==========

# fn_save_bars = os.path.join(cwd, "effectiveness", "effectiveness_comparison_deciduous.png")

# # Create a new dataframe for comparison
# df_comparison = pd.DataFrame(df_CBR['109'])
# df_comparison.columns = ['CBR']
# df_comparison['SIMAREA'] = df_SIMAREA['109']
# df_comparison['CBR_percent'] = df_CBR_percent['109']
# df_comparison['SIMAREA_percent'] = df_SIMAREA_percent['109']
# print(df_comparison)

# # Select a subset of classes
# class_names = ['101', '102']
# long_class_names = ['Agriculture', 'Urban']
# num_classes = np.array([int(x) for x in class_names])
# df_DTF = df_comparison.loc[num_classes]
# # Reset index values and get them for plotting
# df_DTF.reset_index(inplace=True)
# num_classes = df_DTF.index.values
# print(df_DTF)

# # Labels
# r1_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_DTF['CBR'], df_DTF['CBR_percent'])]
# r2_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_DTF['SIMAREA'], df_DTF['SIMAREA_percent'])]
# print(r1_labels)
# print(r2_labels)

# #  Create a barplot
# width = 0.4
# fig, ax = plt.subplots(layout='constrained')
# r1 = ax.bar(num_classes-width/2, df_DTF['CBR'], width, label="Inside")#, log=True)
# r2 = ax.bar(num_classes+width/2, df_DTF['SIMAREA'], width, label="Outside")#, log=True)
# ax.bar_label(r1, r1_labels, padding=3, fontsize=10)
# ax.bar_label(r2, r2_labels, padding=3, fontsize=10)
# ax.set_ylabel("Area of Change (ha)")
# ax.set_xticks(num_classes, long_class_names)
# ax.set_title("Loss from Deciduous Tropical Forest")
# ax.legend(loc='best', ncol=2)
# plt.savefig(fn_save_bars, bbox_inches='tight', dpi=150)

# ========== Create plots for Loss in Evergreen Tropical Forest and Deciduous together(Hectares) ==========

fn_CBR = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table.csv")
fn_SIMAREA = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table.csv")
fn_CBR_percent = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table_percent.csv")
fn_SIMAREA_percent = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table_percent.csv")
fn_save_bars_trop = os.path.join(cwd, "effectiveness", "effectiveness_comparison_tropical_forest.png")
fn_save_bars_built = os.path.join(cwd, "effectiveness", "effectiveness_comparison_built.png")

# Read the percentages
df_CBR = pd.read_csv(fn_CBR, index_col=0)
df_SIMAREA = pd.read_csv(fn_SIMAREA, index_col=0)
df_CBR_percent = pd.read_csv(fn_CBR_percent, index_col=0)
df_SIMAREA_percent = pd.read_csv(fn_SIMAREA_percent, index_col=0)
print(df_CBR)
print(df_SIMAREA.info())

class_names = list(df_CBR.columns)
num_classes = np.array([int(x) for x in class_names])
print(f"Classes: {num_classes}")

# Create a new dataframe for comparison
df_comparison_106 = pd.DataFrame(df_CBR['106'])
df_comparison_106.columns = ['CBR']
df_comparison_106['SIMAREA'] = df_SIMAREA['106']
df_comparison_106['CBR_percent'] = df_CBR_percent['106']
df_comparison_106['SIMAREA_percent'] = df_SIMAREA_percent['106']
print(df_comparison_106)

# Select a subset of classes
# class_names = ['101', '102', '109']
# long_class_names = ['Agriculture', 'Urban', 'Deciduous tropical forest']
class_names = ['101', '102']
long_class_names = ['Agriculture', 'Urban']
num_classes = np.array([int(x) for x in class_names])
df_ETF = df_comparison_106.loc[num_classes]
# Reset index values and get them for plotting
df_ETF.reset_index(inplace=True)
num_classes = df_ETF.index.values
print(df_ETF)

# Labels
r1_labels_106 = [f"{area:,.0f}\n({percent:.2f}%)" for area, percent in zip(df_ETF['CBR'], df_ETF['CBR_percent'])]
r2_labels_106 = [f"{area:,.0f}\n({percent:.2f}%)" for area, percent in zip(df_ETF['SIMAREA'], df_ETF['SIMAREA_percent'])]
print(r1_labels_106)
print(r2_labels_106)

# Create a new dataframe for comparison
df_comparison_109 = pd.DataFrame(df_CBR['109'])
df_comparison_109.columns = ['CBR']
df_comparison_109['SIMAREA'] = df_SIMAREA['109']
df_comparison_109['CBR_percent'] = df_CBR_percent['109']
df_comparison_109['SIMAREA_percent'] = df_SIMAREA_percent['109']
print(df_comparison_109)

#  Create a barplot for 106 evergreen tropical forest
width = 0.4
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), layout='constrained', sharey=True)
r1_106 = ax[0].bar(num_classes-width/2, df_ETF['CBR'], width, label="Inside", color="#ff7f00")#, log=True)
r2_106 = ax[0].bar(num_classes+width/2, df_ETF['SIMAREA'], width, label="Outside", color="#a6cee3")#, log=True)
ax[0].bar_label(r1_106, r1_labels_106, padding=3, fontsize=9)
ax[0].bar_label(r2_106, r2_labels_106, padding=4, fontsize=9)
ax[0].set_ylabel("Area of Change (ha)")
ax[0].set_xticks(num_classes, long_class_names)
ax[0].set_title("Loss from Evergreen Tropical Forest")
# ax[0].set_ylim(0, 30000)
ax[0].set_ylim(0, 22000)
ax[0].legend(loc='upper right', ncol=2)

# Select a subset of classes
# class_names = ['101', '102', '106']
# long_class_names = ['Agriculture', 'Urban', 'Evergreen tropical forest']
class_names = ['101', '102']
long_class_names = ['Agriculture', 'Urban']
num_classes = np.array([int(x) for x in class_names])
df_DTF = df_comparison_109.loc[num_classes]
# Reset index values and get them for plotting
df_DTF.reset_index(inplace=True)
num_classes = df_DTF.index.values
print(df_DTF)

# Labels
r1_labels_109 = [f"{area:,.0f}\n({percent:.2f}%)" for area, percent in zip(df_DTF['CBR'], df_DTF['CBR_percent'])]
r2_labels_109 = [f"{area:,.0f}\n({percent:.2f}%)" for area, percent in zip(df_DTF['SIMAREA'], df_DTF['SIMAREA_percent'])]
print(r1_labels_109)
print(r2_labels_109)

#  Create a barplot for 109 deciduous tropical forest
r1_109 = ax[1].bar(num_classes-width/2, df_DTF['CBR'], width, label="Inside", color="#ff7f00")#, log=True)
r2_109 = ax[1].bar(num_classes+width/2, df_DTF['SIMAREA'], width, label="Outside", color="#a6cee3")#, log=True)
ax[1].bar_label(r1_109, r1_labels_109, padding=3, fontsize=9)
ax[1].bar_label(r2_109, r2_labels_109, padding=4, fontsize=9)
# ax[1].set_ylabel("Area of Change (ha)")
ax[1].set_xticks(num_classes, long_class_names)
ax[1].set_title("Loss from Deciduous Tropical Forest")
ax[1].legend(loc='upper right', ncol=2)
plt.savefig(fn_save_bars_trop, bbox_inches='tight', dpi=150)

# # ========== Create plots for Wetland (Hectares) ==========

# fn_CBR = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table.csv")
# fn_SIMAREA = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table.csv")
# fn_CBR_percent = os.path.join(cwd, "effectiveness", "CBR_land_cover_class_changes_table_percent.csv")
# fn_SIMAREA_percent = os.path.join(cwd, "effectiveness", "SIMAREA_land_cover_class_changes_table_percent.csv")
# fn_save_bars = os.path.join(cwd, "effectiveness", "effectiveness_comparison_wetland.png")

# # Read the percentages
# df_CBR = pd.read_csv(fn_CBR, index_col=0)
# df_SIMAREA = pd.read_csv(fn_SIMAREA, index_col=0)
# df_CBR_percent = pd.read_csv(fn_CBR_percent, index_col=0)
# df_SIMAREA_percent = pd.read_csv(fn_SIMAREA_percent, index_col=0)
# print(df_CBR)
# print(df_SIMAREA.info())

# class_names = list(df_CBR.columns)
# num_classes = np.array([int(x) for x in class_names])
# print(f"Classes: {num_classes}")

# # For the Loss plot
# # # Create a new dataframe for comparison for wetland
# # df_comparison = pd.DataFrame(df_CBR['108'])
# # df_comparison.columns = ['CBR']
# # df_comparison['SIMAREA'] = df_SIMAREA['108']
# # df_comparison['CBR_percent'] = df_CBR_percent['108']
# # df_comparison['SIMAREA_percent'] = df_SIMAREA_percent['108']
# # print(df_comparison)

# # # Select a subset of classes
# # class_names = ['103', '105', '106', '107', '109']
# # long_class_names = ['Water', 'Mangrove', 'Evergreen tropical forest', 'Savanna', 'Deciduous trop. frst.']
# # num_classes = np.array([int(x) for x in class_names])
# # df_wetland = df_comparison.loc[num_classes]
# # # Reset index values and get them for plotting
# # df_wetland.reset_index(inplace=True)
# # num_classes = df_wetland.index.values
# # print(df_wetland)

# # # Labels
# # r1_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_wetland['CBR'], df_wetland['CBR_percent'])]
# # r2_labels = [f"{area:,.0f} ({percent:.2f}%)" for area, percent in zip(df_wetland['SIMAREA'], df_wetland['SIMAREA_percent'])]
# # print(r1_labels)
# # print(r2_labels)

# # #  Create a barplot
# # width = 0.4
# # fig, ax = plt.subplots(layout='constrained')
# # r1 = ax.bar(num_classes-width/2, df_wetland['CBR'], width, label="Inside")#, log=True)
# # r2 = ax.bar(num_classes+width/2, df_wetland['SIMAREA'], width, label="Outside")#, log=True)
# # ax.bar_label(r1, r1_labels, padding=3, fontsize=10)
# # ax.bar_label(r2, r2_labels, padding=3, fontsize=10)
# # ax.set_ylabel("Area of Change (ha)")
# # ax.set_xticks(num_classes, long_class_names)
# # ax.set_title("Loss from Wetland")
# # ax.legend(loc='upper left', ncol=2)
# # plt.savefig(fn_save_bars, bbox_inches='tight', dpi=150)

# # For the Gain plot
# # Create a new dataframe for comparison for wetland
# df_comparison = pd.DataFrame(df_CBR[103, 105, 106, 107, 109])
# # df_comparison.columns = ['CBR']
# df_comparison['SIMAREA'] = df_SIMAREA['103', '105', '106', '107', '109']
# df_comparison['CBR_percent'] = df_CBR_percent['103', '105', '106', '107', '109']
# df_comparison['SIMAREA_percent'] = df_SIMAREA_percent['103', '105', '106', '107', '109']
# df_comparison_transp = df_comparison.transpose()
# print(df_comparison_transp)