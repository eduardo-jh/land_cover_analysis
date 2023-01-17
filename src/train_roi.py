import sys
from datetime import datetime
# adding the directory with modules
# sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
from rsmodule import open_raster, create_raster, land_cover_percentages

# Identify and name each result with its execution time
fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

# fn_landcover = '/vipdata/2023/CALAKMUL/ROI1/TRAIN/usv250s7cw_ROI1_LC_KEY.tif'
fn_landcover = 'D:/Desktop/CALAKMUL/ROI1/training/usv250s7cw_ROI1_LC_KEY.tif'
fn_keys = 'D:/Desktop/CALAKMUL/ROI1/training/Calakmul_keys.txt'
fn_stats = f'D:/Desktop/CALAKMUL/ROI1/results/{datetime.strftime(start, fmt)}_statistics.csv'
fn_plot = f'D:/Desktop/CALAKMUL/ROI1/results/{datetime.strftime(start, fmt)}_plot.png'

# Plot land cover horizontal bar
# plot_hbar(lc_description, percentages, cwd + 'ML/san_juan_gap_lc_dist.png', 'GAP/LANDFIRE Land Cover Classes in San Juan River', 'Percentage (based on pixel count)', (0,30))
land_cover_percentages(fn_landcover, fn_keys, fn_stats,
                    indices=(0,1,2),
                    plot=fn_plot,
                    title='INEGI Land Cover Classes (by group) in Calakmul Biosphere Reserve')
