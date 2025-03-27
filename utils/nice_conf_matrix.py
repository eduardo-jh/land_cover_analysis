#!/usr/bin/env python
# coding: utf-8

""" Plot for the confusion matrices of features in a random forest """

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

# sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

# import rsmodule as rs

# plt.style.use('ggplot')  # R-like plots
plt.style.use("seaborn")
plt.rcParams["font.family"] = "Arial"


def circular_confmatrix(cwd):
    """ Generate a pretty confusion matrix with circles,
        its size indicates the percentage.
    """
    # Open the 2013-2016 labels and predictions
    fn_p1 = os.path.join(cwd, "2013_2016", "results", "2024_03_03-21_05_46", "2024_03_03-21_05_46_confussion_matrix.csv")
    fn_p1_plot = os.path.join(cwd, "2013_2016", "results", "2024_03_03-21_05_46", "2024_03_03-21_05_46_confussion_matrix_pretty.png")
    df_p1 = pd.read_csv(fn_p1, header=None)
    print(df_p1)

    # # Normalize by column
    # df_norm_col = pd.DataFrame({})
    # for i in range(11):
    #     df_norm_col[i] = df_p1[i]/sum(df_p1[i])
    # # print(df_norm_col)

    # Normalize by row
    df_p1_t = df_p1.T # transpose
    df_norm_row = pd.DataFrame({})
    for i in range(11):
        df_norm_row[i] = df_p1_t[i]/sum(df_p1_t[i])
    # print(df_norm_row)

    matrix = df_norm_row
    matrix.columns = [str(i) for i in range(101, 112)]
    print(matrix)

    # create a white grid with the same dimensions as the correlation matrix
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('white')
    ax.imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

    # set the tick labels and rotation for the x and y axes
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.columns)))

    # shift ticks of x axis to top of the graph
    ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

    # format ticks
    ax.set_yticklabels(matrix.columns, fontsize=16, color = "black", fontweight = "normal")
    ax.set_xticklabels(matrix.columns, fontsize=16, color = "black", fontweight = "normal")


    # create grid lines between the tick labels
    ax.set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
    ax.set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
    ax.grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

    # add rectangle around the grid
    rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
    ax.add_patch(rect)

    _cmap = 'viridis_r' # coolwarm_r

    # create circles with radius proportional to the absolute value of correlation
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.columns)):
            correlation = matrix.iat[i, j]
            norm = plt.Normalize(0, 1)  # specify the range of values for the colormap
            sm = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
            color = sm.to_rgba(correlation)

            # Circle
            # circle = Circle((i, j), radius=abs(correlation)/2.5, facecolor=color)
            # ax.add_patch(circle)

            # Rectangle
            size = abs(correlation)/1.5
            rect = Rectangle(xy=(i-size/2, j-size/2), width=size, height=size, facecolor=color)
            ax.add_patch(rect)
            # # For debugging
            # ax.annotate(f"{correlation:0.2f}", (i-size/2, j-size/2), color='b', weight='normal', 
            #         fontsize=4, ha='center', va='center')


    # add color bar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    c_scale = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
    cbar = plt.colorbar(c_scale, ax=ax)
    print(f"Saving: {fn_p1_plot}")
    plt.savefig(fn_p1_plot, bbox_inches='tight', dpi=600)

def portrait_square_confmatrix(matrix_files):

    """ Portrait figure containg all confusion matrices """

    # create a white grid with the same dimensions as the correlation matrix
    fig, ax = plt.subplots(2, 3, figsize=(24,12))


    for col, _file in enumerate(matrix_files):
        print(f"Plotting from data: {_file}")

        df = pd.read_csv(fn_p1, header=None)

        # Normalize by column
        df_norm_col = pd.DataFrame({})
        for i in range(11):
            df_norm_col[i] = df[i]/sum(df[i])
        # print(df_norm_col)

        # Normalize by row
        df_trans = df.transpose(copy=True) # transpose
        df_norm_row_temp = pd.DataFrame({})
        for i in range(11):
            df_norm_row_temp[i] = df_trans[i]/sum(df_trans[i])
        # Transpose back
        df_norm_row = df_norm_row_temp.transpose(copy=True)

        #### Top row
        matrix = df_norm_col
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("User's Accuracy (Normalized by columns)")
        print(matrix)

        ax[0,col].set_facecolor('white')
        ax[0,col].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[0,col].set_xticks(np.arange(len(matrix.columns)))
        ax[0,col].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[0,col].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[0,col].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        # ax[0,col].set_xticklabels(matrix.columns, fontsize=16, color = "white", fontweight = "normal", rotation = 90)
        ax[0,col].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[0,col].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)


        # create grid lines between the tick labels
        ax[0,col].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[0,col].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[0,col].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[0,col].add_patch(rect)

        if col == 0:
            ax[0,col].set_ylabel("User's Accuracy", fontsize=16)
            ax[0,col].set_title("P1", fontsize=16)
        if col == 1:
            ax[0,col].set_title("P2", fontsize=16)
        if col == 2:
            ax[0,col].set_title("P3", fontsize=16)

        _cmap = 'viridis_r' # coolwarm_r

        # create squares proportional to the absolute value of correlation
        for i in range(len(matrix.columns)):
            for j in range(len(matrix.columns)):
                correlation = matrix.iat[i, j]
                norm = plt.Normalize(0, 1)  # specify the range of values for the colormap
                sm = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
                color = sm.to_rgba(correlation)

                # Rectangle
                size = abs(correlation)/1.5
                rect = Rectangle(xy=(i-size/2, j-size/2), width=size, height=size, facecolor=color)
                ax[0,col].add_patch(rect)
        
        ##### Lower row
        matrix = df_norm_row.T
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("Producer's Accuracy (Normalized by rows) THIS IS TRANSPOSED")
        print(matrix)

        ax[1,col].set_facecolor('white')
        ax[1,col].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[1,col].set_xticks(np.arange(len(matrix.columns)))
        ax[1,col].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[1,col].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[1,col].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        ax[1,col].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[1,col].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[1,col].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[1,col].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[1,col].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[1,col].add_patch(rect)

        if col == 0:
            ax[1,col].set_ylabel("Producer's Accuracy", fontsize=16)

        _cmap = 'viridis_r' # coolwarm_r

        # create circles with radius proportional to the absolute value of correlation
        for i in range(len(matrix.columns)):
            for j in range(len(matrix.columns)):
                correlation = matrix.iat[i, j]
                norm = plt.Normalize(0, 1)  # specify the range of values for the colormap
                sm = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
                color = sm.to_rgba(correlation)

                # Rectangle
                size = abs(correlation)/1.5
                rect = Rectangle(xy=(i-size/2, j-size/2), width=size, height=size, facecolor=color)
                ax[1,col].add_patch(rect)

    # add color bar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    c_scale = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
    cbar = plt.colorbar(c_scale, ax=ax)
    fn_plot = os.path.join(cwd, "confussion_matrix_plot.png")
    print(f"Saving: {fn_plot}")
    plt.savefig(fn_plot, bbox_inches='tight', dpi=150)

def portrait_grayscale_confmatrix_orig(matrix_files):

    """ Create a portrait array of confusion matrices """

    # create a white grid with the same dimensions as the correlation matrix
    fig, ax = plt.subplots(3, 2, figsize=(12,12), sharex=True, sharey=True)

    for row, _file in enumerate(matrix_files):
        print(f"Plotting from data: {_file}")

        df = pd.read_csv(fn_p1, header=None)

        # Normalize by column
        df_norm_col = pd.DataFrame({})
        for i in range(11):
            df_norm_col[i] = df[i]/sum(df[i])
        # print(df_norm_col)

        # Normalize by row
        df_trans = df.transpose(copy=True) # transpose
        df_norm_row_temp = pd.DataFrame({})
        for i in range(11):
            df_norm_row_temp[i] = df_trans[i]/sum(df_trans[i])
        # Transpose back
        df_norm_row = df_norm_row_temp.transpose(copy=True)

        #### First column
        matrix = df_norm_col
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("User's Accuracy (Normalized by columns)")
        print(matrix)

        ax[row,0].set_facecolor('white')
        ax[row,0].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[row,0].set_xticks(np.arange(len(matrix.columns)))
        ax[row,0].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[row,0].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[row,0].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        # ax[row,0].set_xticklabels(matrix.columns, fontsize=16, color = "white", fontweight = "normal", rotation = 90)
        ax[row,0].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[row,0].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[row,0].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[row,0].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[row,0].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[row,0].add_patch(rect)

        if row == 0:
            ax[row,0].set_ylabel("a) P1", fontsize=16, rotation='horizontal')
            ax[row,0].set_title("User's Accuracy", fontsize=16)
        if row == 1:
            ax[row,0].set_ylabel("b) P2", fontsize=16, rotation='horizontal')
        if row == 2:
            ax[row,0].set_ylabel("c) P3", fontsize=16, rotation='horizontal')

        _cmap = 'gray_r'

        ax[row,0].imshow(matrix.T, cmap=_cmap)
        ax[row,0].grid(False)
        ax[row,0].set_aspect('equal')
        
        ##### Second column
        matrix = df_norm_row.T
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("Producer's Accuracy (Normalized by rows) THIS IS TRANSPOSED")
        print(matrix)

        ax[row,1].set_facecolor('white')
        ax[row,1].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[row,1].set_xticks(np.arange(len(matrix.columns)))
        ax[row,1].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[row,col].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[row,col].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        # ax[row,1].set_yticklabels([])
        # ax[row,1].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[row,1].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[row,1].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[row,1].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[row,1].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[row,1].add_patch(rect)

        if row == 0:
            ax[row,1].set_title("Producer's Accuracy", fontsize=16)

        ax[row,1].imshow(matrix.T, cmap=_cmap)
        ax[row,1].grid(False)
        ax[row,1].set_aspect('equal')

    # make_axes_area_auto_adjustable(ax)
    plt.subplots_adjust(wspace=0, hspace=0.01)

    # add color bar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    c_scale = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
    cbar = plt.colorbar(c_scale, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    fn_plot = os.path.join(cwd, "confussion_matrix_plot_gray_300dpi.jpg")
    print(f"Saving: {fn_plot}")
    plt.savefig(fn_plot, bbox_inches='tight', dpi=300)


def portrait_grayscale_confmatrix(matrix_files):

    """ Create a portrait array of confusion matrices """

    # create a white grid with the same dimensions as the correlation matrix
    fig, ax = plt.subplots(3, 2, figsize=(12,12), sharex=True, sharey=True)

    for row, _file in enumerate(matrix_files):
        print(f"Plotting from data: {_file}")

        df = pd.read_csv(fn_p1, header=None)

        # Normalize by column
        df_norm_col = pd.DataFrame({})
        for i in range(11):
            df_norm_col[i] = df[i]/sum(df[i])
        # print(df_norm_col)

        # Normalize by row
        df_trans = df.transpose(copy=True) # transpose
        df_norm_row_temp = pd.DataFrame({})
        for i in range(11):
            df_norm_row_temp[i] = df_trans[i]/sum(df_trans[i])
        # Transpose back
        df_norm_row = df_norm_row_temp.transpose(copy=True)

        #### First column
        matrix = df_norm_col
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("User's Accuracy (Normalized by columns)")
        print(matrix)

        ax[row,0].set_facecolor('white')
        ax[row,0].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[row,0].set_xticks(np.arange(len(matrix.columns)))
        ax[row,0].set_yticks(np.arange(len(matrix.columns)))

        ax[row,0].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[row,0].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[row,0].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[row,0].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[row,0].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[row,0].add_patch(rect)

        if row == 0:
            ax[row,0].set_ylabel("P1", fontsize=16)
            # ax[row,0].set_ylabel("a)\nP1", fontsize=16, rotation='horizontal')
            ax[row,0].set_title("User's Accuracy", fontsize=16)
        if row == 1:
            # ax[row,0].set_ylabel("b)\nP2", fontsize=16, rotation='horizontal')
            ax[row,0].set_ylabel("P2", fontsize=16)
        if row == 2:
            # ax[row,0].set_ylabel("c)\nP3", fontsize=16, rotation='horizontal')
            ax[row,0].set_ylabel("P3", fontsize=16)

        _cmap = 'gray_r'

        ax[row,0].imshow(matrix.T, cmap=_cmap)
        ax[row,0].grid(False)
        ax[row,0].set_aspect('equal')
        
        ##### Second column
        matrix = df_norm_row.T
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("Producer's Accuracy (Normalized by rows) THIS IS TRANSPOSED")
        print(matrix)

        ax[row,1].set_facecolor('white')
        ax[row,1].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[row,1].set_xticks(np.arange(len(matrix.columns)))
        ax[row,1].set_yticks(np.arange(len(matrix.columns)))

        ax[row,1].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[row,1].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[row,1].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[row,1].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[row,1].add_patch(rect)

        if row == 0:
            ax[row,1].set_title("Producer's Accuracy", fontsize=16)

        ax[row,1].imshow(matrix.T, cmap=_cmap)
        ax[row,1].grid(False)
        # ax[row,1].set_aspect('equal')

    # make_axes_area_auto_adjustable(ax)
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()

    # add color bar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    c_scale = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
    cbar = plt.colorbar(c_scale, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    fn_plot = os.path.join(cwd, "confussion_matrix_plot_gray_300dpi.jpg")
    print(f"Saving: {fn_plot}")
    plt.savefig(fn_plot, bbox_inches='tight', dpi=300)


def landscape_square_confmatrix(matrix_files):

    """ Landscape figure containing all confusion matrices"""

    # create a white grid with the same dimensions as the correlation matrix
    fig, ax = plt.subplots(2, 3, figsize=(24,12))

    for col, _file in enumerate(matrix_files):
        print(f"Plotting from data: {_file}")

        df = pd.read_csv(fn_p1, header=None)

        # Normalize by column
        df_norm_col = pd.DataFrame({})
        for i in range(11):
            df_norm_col[i] = df[i]/sum(df[i])
        # print(df_norm_col)

        # Normalize by row
        df_trans = df.transpose(copy=True) # transpose
        df_norm_row_temp = pd.DataFrame({})
        for i in range(11):
            df_norm_row_temp[i] = df_trans[i]/sum(df_trans[i])
        # Transpose back
        df_norm_row = df_norm_row_temp.transpose(copy=True)

        #### Top row
        matrix = df_norm_col
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("User's Accuracy (Normalized by columns)")
        print(matrix)

        ax[0,col].set_facecolor('white')
        ax[0,col].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[0,col].set_xticks(np.arange(len(matrix.columns)))
        ax[0,col].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[0,col].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[0,col].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        # ax[0,col].set_xticklabels(matrix.columns, fontsize=16, color = "white", fontweight = "normal", rotation = 90)
        ax[0,col].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[0,col].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)


        # create grid lines between the tick labels
        ax[0,col].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[0,col].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[0,col].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[0,col].add_patch(rect)

        if col == 0:
            ax[0,col].set_ylabel("User's Accuracy", fontsize=16)
            ax[0,col].set_title("a)\tP1", fontsize=16, rotation='horizontal')
        if col == 1:
            # ax[0,col].set_title("P2", fontsize=16)
            ax[0,col].set_title("b)\tP2", fontsize=16, rotation='horizontal')
        if col == 2:
            ax[0,col].set_title("c)\tP3", fontsize=16, rotation='horizontal')

        _cmap = 'viridis_r' # coolwarm_r

        # create circles with radius proportional to the absolute value of correlation
        for i in range(len(matrix.columns)):
            for j in range(len(matrix.columns)):
                correlation = matrix.iat[i, j]
                norm = plt.Normalize(0, 1)  # specify the range of values for the colormap
                sm = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
                color = sm.to_rgba(correlation)

                # Rectangle
                size = abs(correlation)/1.5
                rect = Rectangle(xy=(i-size/2, j-size/2), width=size, height=size, facecolor=color)
                ax[0,col].add_patch(rect)
        
        ##### Lower row
        matrix = df_norm_row.T
        matrix.columns = [str(i) for i in range(101, 112)]  # land cover labels
        print("Producer's Accuracy (Normalized by rows) THIS IS TRANSPOSED")
        print(matrix)

        ax[1,col].set_facecolor('white')
        ax[1,col].imshow(np.ones_like(matrix), cmap='gray_r', interpolation='nearest')

        # set the tick labels and rotation for the x and y axes
        ax[1,col].set_xticks(np.arange(len(matrix.columns)))
        ax[1,col].set_yticks(np.arange(len(matrix.columns)))

        # shift ticks of x axis to top of the graph
        # ax[1,col].tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)

        # format ticks
        # color = "black"
        # if col != 0:
        #     color = "white"  # hide labels
        # ax[1,col].set_yticklabels(matrix.columns, fontsize=16, color = color, fontweight = "normal")
        ax[1,col].set_yticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal")
        ax[1,col].set_xticklabels(matrix.columns, fontsize=12, color = "black", fontweight = "normal", rotation = 90)

        # create grid lines between the tick labels
        ax[1,col].set_xticks(np.arange(len(matrix.columns) + 1) - .5, minor=True, linestyle="solid")
        ax[1,col].set_yticks(np.arange(len(matrix.columns) + 1) - .5, minor=True,  linestyle="solid")
        ax[1,col].grid(which="minor", color="lightgray", linestyle="solid", linewidth=1, )

        # add rectangle around the grid
        rect = plt.Rectangle((-.5, -.5), len(matrix.columns), len(matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
        ax[1,col].add_patch(rect)

        if col == 0:
            ax[1,col].set_ylabel("Producer's Accuracy", fontsize=16)

        _cmap = 'viridis_r' # coolwarm_r

        # create circles with radius proportional to the absolute value of correlation
        for i in range(len(matrix.columns)):
            for j in range(len(matrix.columns)):
                correlation = matrix.iat[i, j]
                norm = plt.Normalize(0, 1)  # specify the range of values for the colormap
                sm = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
                color = sm.to_rgba(correlation)

                # Rectangle
                size = abs(correlation)/1.5
                rect = Rectangle(xy=(i-size/2, j-size/2), width=size, height=size, facecolor=color)
                ax[1,col].add_patch(rect)

    # add color bar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    c_scale = plt.cm.ScalarMappable(norm=norm, cmap=_cmap)
    cbar = plt.colorbar(c_scale, ax=ax)
    fn_plot = os.path.join(cwd, "confussion_matrix_plot_landscape.png")
    print(f"Saving: {fn_plot}")
    plt.savefig(fn_plot, bbox_inches='tight', dpi=150)


if __name__ == "__main__":

    cwd = "/VIP/engr-didan02s/DATA/EDUARDO/2024/YUCATAN_LAND_COVER/ROI2/"

    # Do a figure containing all periods
    fn_p1 = os.path.join(cwd, "2013_2016", "results", "2024_03_06-23_19_43", "2024_03_06-23_19_43_confussion_matrix.csv")
    fn_p2 = os.path.join(cwd, "2016_2019", "results", "2024_03_08-13_29_31", "2024_03_08-13_29_31_confussion_matrix.csv")
    fn_p3 = os.path.join(cwd, "2019_2022", "results", "2024_03_12-19_32_01", "2024_03_12-19_32_01_confussion_matrix.csv")

    matrix_files = [fn_p1, fn_p2, fn_p3]

    # Create a grayscale array of confusion matrices
    portrait_grayscale_confmatrix(matrix_files)