#!/usr/bin/env python
# coding: utf-8

""" Plot for the importance of features in a random forest """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('ggplot')  # R-like plots
plt.style.use("seaborn")

# cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/results/'
# # fn_importance = cwd + '2024_02_15-19_09_53/2024_02_15-19_09_53_feat_importance.csv'
# fn_importance = cwd + '2023_10_28-01_04_42/2023_10_28-01_04_42_feat_importance.csv'

# fn_save_imp_fig = '/data/ssd/eduardojh/results/test_feat_importance.png'

# df = pd.read_csv(fn_importance)
# df.sort_values(by='Importance', ascending=True, inplace=True)
# # print(df)

# # Save a figure with the importances and error bars
# # fig, ax = plt.subplots()
# # ax.barh(df['Feature'], df['Importance'])
# # ax.set_title("Feature importances")
# # ax.set_ylabel("Mean decrease in impurity")
# # fig.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=300)

# plt.figure(figsize=(8, 16), constrained_layout=True)
# plt.barh(df['Feature'], df['Importance'])
# plt.title("Feature importances")
# plt.ylabel("Mean decrease in impurity")
# plt.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=300)



# ========== A feature importance plot with the three periods together ==========

fn_p1 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/results/2024_02_29-09_37_28/2024_02_29-09_37_28_feat_importance.csv"
fn_p2 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/results/2024_03_01-11_34_45/2024_03_01-11_34_45_feat_importance.csv"
fn_p3 = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/results/2024_03_04-23_10_16/2024_03_04-23_10_16_feat_importance.csv"

df_p1 = pd.read_csv(fn_p1, index_col=False)
df_p2 = pd.read_csv(fn_p2, index_col=False)
df_p3 = pd.read_csv(fn_p3, index_col=False)

fn_save_imp_fig = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/feature_importance_all_periods_40feats.png"

# print(df_p1)
df_p1 = df_p1[['Feature', 'Importance', 'Stdev']]
df_p1.columns = ['Feature', 'Importance_P1', 'Stdev_P1']
df_p1 = df_p1.reset_index(drop=True)

df_p2 = df_p2[['Feature', 'Importance', 'Stdev']]
df_p2.columns = ['Feature', 'Importance_P2', 'Stdev_P2']
df_p2 = df_p2.reset_index(drop=True)

df_p3 = df_p3[['Feature', 'Importance', 'Stdev']]
df_p3.columns = ['Feature', 'Importance_P3', 'Stdev_P3']
df_p3 = df_p3.reset_index(drop=True)

df = pd.merge(df_p1, df_p2, on='Feature', how='outer')
df = pd.merge(df, df_p3, on='Feature', how='outer')
print(df)

y = np.arange(1, len(df['Feature'])+1, 1)
h = 0.3

print("Creating a horizontal bar plot")

f, ax = plt.subplots(figsize=(12, 24))
ax.barh(y-h, df['Importance_P1'], height=h, xerr=df['Stdev_P1'], label='P1')
ax.barh(y, df['Importance_P2'], height=h, xerr=df['Stdev_P2'], label='P2')
ax.barh(y+h, df['Importance_P3'], height=h, xerr=df['Stdev_P3'], label='P3')
ax.set_ylim(0,41)
ax.set_title("Feature importances")
ax.set_xlabel("Mean decrease in impurity")
ax.set_yticks(y)
ax.set_yticklabels(df['Feature'])
ax.grid(axis='y')
ax.legend(loc='lower right')
plt.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=600)

print("Creating a bar plot")

f, ax = plt.subplots(figsize=(24, 12))
ax.bar(y-h, df['Importance_P1'], width=h, yerr=df['Stdev_P1'], label='P1')
ax.bar(y, df['Importance_P2'], width=h, yerr=df['Stdev_P2'], label='P2')
ax.bar(y+h, df['Importance_P3'], width=h, yerr=df['Stdev_P3'], label='P3')
ax.set_xlim(0,41)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
ax.invert_xaxis()
ax.set_xticks(y)
ax.set_xticklabels(df['Feature'], rotation=90)
ax.grid(axis='x')
ax.legend(loc='upper right')
plt.savefig(fn_save_imp_fig[:-4] + "_vert.png", bbox_inches='tight', dpi=600)

## Another approach, but no possible to set standard deviation

# df_p1 = df_p1[['Feature', 'Importance', 'Stdev']]
# df_p1['Period'] = ['P1'] * len(df_p1['Feature'])
# df_p1 = df_p1.reset_index(drop=True)
# print(df_p1)

# df_p2 = df_p2[['Feature', 'Importance', 'Stdev']]
# df_p2['Period'] = ['P2'] * len(df_p2['Feature'])
# df_p2 = df_p2.reset_index(drop=True)

# df_p3 = df_p3[['Feature', 'Importance', 'Stdev']]
# df_p3['Period'] = ['P3'] * len(df_p3['Feature'])
# df_p3 = df_p3.reset_index(drop=True)

# df = df_p1.append(df_p2)
# df = df.append(df_p3)
# df = df.reset_index(drop=True)
# print(df)

# sns.barplot(df, x="Feature", y="Importance", hue="Period", yerr='Stdev')
# plt.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=600)

print("All done! ;-)")