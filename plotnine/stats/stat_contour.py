from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pandas as pd
import matplotlib._tri as _tri
from matplotlib.tri.triangulation import Triangulation
from mizani.breaks import extended_breaks

from ..doctools import document
from .stat import stat

@document
class stat_contour(stat):
    '''
    Compute 2D contour areas

    {usage}

    Parameters:
    -----------
    {common_parameters}
    levels: int or array_like
        Contour levels. If an integer, it specifies the maximum number
        of levels, if array_like it is the levels themselves. Default
        is 5.
    '''
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'contour', 'position': 'identity',
                      'na_rm': False, 'levels': 5, 'levelwidth': None}

    @classmethod
    def compute_group(cls, data, scales, **params):
        group = data['group'].iloc[0]

        #print('cls: %s'%(str(cls)))
        #print('data: %s'%(str(scales)))
        #print('scales: %s'%(str(scales)))
        #print('params: %s'%(str(params)))

        data = contour_filled_tri(
            data['x'].values,
            data['y'].values,
            data['fill'].values,
            params['levels'])

        groups = (str(group) + '-' + data['fill'].astype(str))
        data['group'] = groups

        return data

def contour_filled_tri(X, Y, Z, levels):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    zmin, zmax = Z.min(), Z.max()

    triangulation = Triangulation(X, Y, triangles=None, mask=None)

    contour_generator = _tri.TriContourGenerator(
        triangulation.get_cpp_triangulation(), Z)

    if isinstance(levels, int):
        levels = extended_breaks(n=levels)((zmin, zmax))

    all_segments = []
    all_levels = []
    all_seg_codes = []

    for i in range(len(levels) - 1):
        # The contour_generator.create_filled_contour returns a 2D array 
        # (npts, 2) of point coordinates corresponding to segments of contour 
        # lines. There are contour lines at one level and contour lines at the
        # next level.
        # seg_code is a (npts, ) 1D array of 1s and 2s. A 1 corresponds to the
        # start of a new level line, and a 2 corresponds to a point belonging
        # to that line.
        segs, seg_codes = contour_generator.create_filled_contour(
            levels[i], levels[i+1])
        npts = segs.shape[0]
        all_segments.append(segs)
        all_levels.append(np.repeat(levels[i], npts))
        all_seg_codes.append(seg_codes)

    x, y = np.vstack(all_segments).T
    level = np.hstack(all_levels)
    seg_codes = np.hstack(all_seg_codes)

    data = pd.DataFrame({
        'x': x,
        'y': y,
        'fill': level,
        'seg_codes': seg_codes
    })

    return data

def contour_lines_tri(X, Y, Z, levels):
    '''
    Compute contour lines based on triangulation
    '''
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    zmin, zmax = Z.min(), Z.max()

    triangulation = Triangulation(X, Y)
    contour_generator = _tri.TriContourGenerator(
        triangulation.get_cpp_triangulation, Z)
    