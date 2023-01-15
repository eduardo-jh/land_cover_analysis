#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_mosaic_roi(mosaic, roi, **kwargs):
    _savefigs = kwargs.get('savefigs', '')
    roi_ulx, roi_uly, roi_lrx,roi_lry = roi

    fig, ax = plt.subplots()
    ax.plot(roi_ulx, roi_uly, 'ro')
    ax.add_patch(Rectangle((roi_ulx, roi_lry), roi_lrx-roi_ulx, roi_uly-roi_lry, fc='none', edgecolor='b'))
    print(f'x={roi_ulx},y={roi_lry} width={roi_lrx-roi_ulx}, height={roi_uly-roi_lry}')
                        
    for scene_id in mosaic_extent.keys():
        sce_ulx, sce_lrx, sce_uly, sce_lry = mosaic_extent[scene_id]
        ax.add_patch(Rectangle((sce_ulx, sce_lry), sce_lrx-sce_ulx, sce_uly-sce_lry, fc='none', edgecolor='k', linewidth=1))

    if _savefigs != '':
        plt.savefig(f'{_savefigs}_extents.png', bbox_inches='tight', dpi=600)
        plt.close()


if __name__ == '__main__':
    roi_ulx = round(30442.2081462452188134)
    roi_uly = round(2180910.3759473729878664)
    roi_lrx = round(395043.8311359137296677)
    roi_lry = round(1882497.1240605544298887)
    
    mosaic_extent = {'019046': [287685.0, 513615.0, 2347215.0, 2126385.0], '019047': [251685.0, 477615.0, 2187615.0, 1967085.0], '019048': [215385.0, 441615.0, 2028315.0, 1807485.0], '020046': [124785.0, 352215.0, 2349615.0, 2127585.0], '020047': [87285.0, 314715.0, 2190315.0, 1968285.0], '020048': [48840.0, 278550.0, 2032230.0, 1807740.0], '021046': [-38970.0, 192750.0, 2354970.0, 2128470.0], '021047': [-78030.0, 153900.0, 2195820.0, 1969710.0], '021048': [-116700.0, 114840.0, 2036490.0, 1810470.0]}

    show_mosaic_roi(mosaic_extent, [roi_ulx, roi_uly, roi_lrx,roi_lry], savefigs='/home/eduardojh/Documents/results/example_rectangles')

