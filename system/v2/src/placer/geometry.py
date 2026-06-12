"""Small shared geometry helpers for placement math."""

import numpy as np


def separation_matrices(sizes: np.ndarray):
    """Pairwise minimum center separations for non-overlap, as two [n, n] grids.

    ``sep_*[i, j]`` is the half-sum of macro i and j's extent on that axis: two
    macros overlap iff ``|cx_i - cx_j| < sep_x[i, j]`` and likewise on y. Used by
    the legalizer and the 2-opt / relocation passes for conflict checks.
    """
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2  # [n, n]
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    return sep_x_mat, sep_y_mat
