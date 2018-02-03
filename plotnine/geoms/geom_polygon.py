from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.path as mpath
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

from ..utils import to_rgba, SIZE_FACTOR
from ..doctools import document
from .geom import geom


@document
class geom_polygon(geom):
    """
    Polygon, a filled path

    {usage}

    Parameters
    ----------
    {common_parameters}

    Notes
    -----
    All paths in the same ``group`` aesthetic value make up a polygon.
    """
    DEFAULT_AES = {'alpha': 1, 'color': None, 'fill': '#333333',
                   'linetype': 'solid', 'size': 0.5}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity',
                      'na_rm': False}
    REQUIRED_AES = {'x', 'y'}

    def handle_na(self, data):
        return data

    def draw_panel(self, data, panel_params, coord, ax, **params):
        """
        Plot all groups
        """
        self.draw_group(data, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data, panel_params, coord, ax, **params):
        data = coord.transform(data, panel_params, munch=True)
        data['size'] *= SIZE_FACTOR

        # Some stats may order the data in ways that prevent
        # objects from occluding other objects. We do not want
        # to undo that order.
        grouper = data.groupby('group', sort=False)
        for i, (group, df) in enumerate(grouper):
            segs = np.column_stack([df['x'], df['y']])
            nsegs = segs.shape[0]
            if 'seg_codes' in df:
                codes = df['seg_codes']
            else:
                codes = [1] + ([2] * (nsegs - 1))
            segs, codes = close_curves(segs, codes)

            fill = to_rgba(df['fill'].iloc[0], df['alpha'].iloc[0])
            facecolor = 'none' if fill is None else fill
            edgecolor = df['color'].iloc[0] or 'none'
            linestyle = df['linetype'].iloc[0]
            linewidth = df['size'].iloc[0]

            paths = [mpath.Path(segs, codes=codes)]

            col = PathCollection(
                paths,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linestyles=linestyle,
                linewidths=linewidth,
                transOffset=ax.transData,
                zorder=params['zorder'])

            ax.add_collection(col)

    @staticmethod
    def draw_legend(data, da, lyr):
        """
        Draw a rectangle in the box

        Parameters
        ----------
        data : dataframe
        da : DrawingArea
        lyr : layer

        Returns
        -------
        out : DrawingArea
        """
        data['size'] *= SIZE_FACTOR
        # We take into account that the linewidth
        # bestrides the boundary of the rectangle
        linewidth = np.min([data['size'],
                            da.width/4, da.height/4])
        if data['color'] is None:
            linewidth = 0

        facecolor = to_rgba(data['fill'], data['alpha'])
        if facecolor is None:
            facecolor = 'none'

        rect = Rectangle((0+linewidth/2, 0+linewidth/2),
                         width=da.width-linewidth,
                         height=da.height-linewidth,
                         linewidth=linewidth,
                         linestyle=data['linetype'],
                         facecolor=facecolor,
                         edgecolor=data['color'],
                         capstyle='projecting')
        da.add_artist(rect)
        return da

def close_curves(segs, codes):
    segs = np.asarray(segs, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int8)
    npts = segs.shape[0]
    discs = np.argwhere(codes == 1).flatten().tolist() + [npts]
    ndiscs = len(discs) - 1
    npts_repeated = npts + ndiscs
    repeated_segs = np.empty((npts_repeated, 2), dtype=np.float64)
    repeated_codes = np.empty((npts_repeated, ), dtype=np.int8)

    for i_disc, (i_lo, i_hi) in enumerate(zip(discs[0:-1], discs[1:])):
        irpt_lo = i_lo + i_disc
        irpt_hi = i_hi + i_disc
        repeated_segs[irpt_lo:irpt_hi, :] = segs[i_lo:i_hi, :]
        repeated_segs[irpt_hi, :] = segs[i_lo, :]
        repeated_codes[irpt_lo] = mpath.Path.MOVETO  # = 1
        repeated_codes[irpt_lo + 1:irpt_hi] = mpath.Path.LINETO  # = 2
        repeated_codes[irpt_hi] = mpath.Path.CLOSEPOLY  # = 79

    return repeated_segs, repeated_codes