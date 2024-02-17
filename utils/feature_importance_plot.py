#!/usr/bin/env python
# coding: utf-8

""" Plot for the importance of features in a random fores """

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # R-like plots

cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/results/'
# fn_importance = cwd + '2024_02_15-19_09_53/2024_02_15-19_09_53_feat_importance.csv'
fn_importance = cwd + '2023_10_28-01_04_42/2023_10_28-01_04_42_feat_importance.csv'

fn_save_imp_fig = '/data/ssd/eduardojh/results/test_feat_importance.png'

df = pd.read_csv(fn_importance)
df.sort_values(by='Importance', ascending=True, inplace=True)
# print(df)

# Save a figure with the importances and error bars
# fig, ax = plt.subplots()
# ax.barh(df['Feature'], df['Importance'])
# ax.set_title("Feature importances")
# ax.set_ylabel("Mean decrease in impurity")
# fig.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=300)

plt.figure(figsize=(8, 16), constrained_layout=True)
plt.barh(df['Feature'], df['Importance'])
plt.title("Feature importances")
plt.ylabel("Mean decrease in impurity")
plt.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=300)