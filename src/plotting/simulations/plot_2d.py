"""
Functions for plotting 2D simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

def plot_pressure_2d(
    pressure_function,
    time_days,
    figures_directory,
    nx=100,
    ny=50,
):
    """
    Plots the 2D pressure field using a regular grid.

    Parameters
    ----------
    pressure_function : Firedrake Function
        Pressure field to be plotted.

    time_days : float
        Simulation time in days.

    figures_directory : Path
        Directory where the figure will be saved.

    nx : int
        Number of points in the X direction.

    ny : int
        Number of points in the Y direction.
    """

    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 0.2, ny)

    XX, YY = np.meshgrid(x_grid, y_grid)

    pressure_grid = sample_field(
        pressure_function,
        XX,
        YY,
    )

    figure, axis = plt.subplots(figsize=(10, 4))

    contour = axis.contourf(
        XX,
        YY,
        pressure_grid,
        levels=20,
    )

    colorbar = figure.colorbar(contour, ax=axis)
    colorbar.set_label("Pressão adimensional")

    axis.set_xlabel("X")
    axis.set_ylabel("Y")

    axis.set_title(
        f"Campo de pressão — t = {time_days:.1f} dias"
    )

    axis.set_aspect("equal")

    figure.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(parents=True, exist_ok=True)

    filename = (
        figures_directory
        / f"pressure_2d_{time_days:.1f}_days.png"
    )

    figure.savefig(filename, dpi=300)
    plt.close(figure)