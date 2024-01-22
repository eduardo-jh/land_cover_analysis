#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Generate a shapefile with points of the INFyS

WARNING: This script only genetates the CSV file, a shapefile should be generated in QGIS
by using the X, Y columns!

Last edited: 2024-01-22
"""

import os
import pandas as pd

cwd = "/vipdata/2023/GIS_Calakmul/Data/"

# src_fn = os.path.join(cwd, "INFYS_ROI2.csv")  # Selected from "Composicion y Estructura"
# src_fn = os.path.join(cwd, "INVENTARIO_FORESTAL", "INFYS_Arbolado_2015_2020.xlsx")

# # df = pd.read_csv(src_fn)
# df = pd.read_excel(src_fn)
# print(df)
# print(df.info())

# # Select the rows for the Yucatan Peninsula: Yucatan, Campeche, Quintana Roo, Tabasco
# df1 = df[df["Estado_C3"].isin(['Tabasco', 'Campeche', 'Quintana Roo', 'Yucatán', 'Yucatan'])]
# print(df1)
# print(df1.info())

# # Select a subgroup of columns only
# columns = ["UPMID", "IdConglomerado", "Anio_C3", "Estado_C3", "CVE_S7_C3", "X_C3", "Y_C3", "DESCRIP_S7_C3"]
# df1 = df1[columns]
# df1 = df1.reset_index(drop=True)
# print(df1)
# print(df1.info())

# # Save to a CSV file
# df1.to_csv(src_fn[:-5] + "_ROI2.csv")

###### Second part #####

src_fn = os.path.join(cwd, "INVENTARIO_FORESTAL", "INFYS_Arbolado_2015_2020_ROI2.csv")
df = pd.read_csv(src_fn, index_col=None)
print(df)

# Comprare 'UPMID' and 'IdConglomerado' keys to see if they are unique
df1 = df.groupby(['UPMID']).first()
print(df1)
print(df1.info())
df1.to_csv(src_fn[:-4] + "_group1.csv")

df2 = df.groupby(['IdConglomerado']).first()
print(df2)
print(df2.info())
df2.to_csv(src_fn[:-4] + "_group2.csv")

# After comparison both keys are unique, either one can be used to generate a shapefile
