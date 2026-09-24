"""
Functions for plotting 2D simulation results.
"""

import numpy as np


def point_value(function, x_value, y_value, Lx=1.0, Ly=0.2):
    """
    Returns the value of a Firedrake function at a point.
    """

    eps = 1.0e-10

    x_clamped = min(max(float(x_value), eps), Lx - eps)
    y_clamped = min(max(float(y_value), eps), Ly - eps)

    return float(function.at((x_clamped, y_clamped)))


def sample_field(function, x_grid, y_grid, Lx=1.0, Ly=0.2):
    """
    Samples a scalar Firedrake function on a regular grid.

    Parameters
    ----------
    function : Firedrake Function
        Scalar field to be sampled.

    x_grid : numpy.ndarray
        X coordinates of the sampling grid.

    y_grid : numpy.ndarray
        Y coordinates of the sampling grid.

    Lx : float
        Domain length in the X direction.

    Ly : float
        Domain length in the Y direction.

    Returns
    -------
    numpy.ndarray
        Field values on the regular grid.
    """

    values = np.zeros_like(x_grid, dtype=float)

    for i in range(y_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            values[i, j] = point_value(
                function,
                x_grid[i, j],
                y_grid[i, j],
                Lx,
                Ly,
            )

    return values