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

    print("Shape da pressão:", pressure_grid.shape)
    print("Menor valor:", np.nanmin(pressure_grid))
    print("Maior valor:", np.nanmax(pressure_grid))
    print("Quantidade de NaN:", np.isnan(pressure_grid).sum())
    print("Quantidade de valores finitos:", np.isfinite(pressure_grid).sum())

    figure, axis = plt.subplots(figsize=(10, 4))

    pressure_min = pressure_grid.min()
    pressure_max = pressure_grid.max()

    levels = np.linspace(
        pressure_min - 1.0e-6,
        pressure_max + 1.0e-6,
        21,
    )

    contour = axis.contourf(
        XX,
        YY,
        pressure_grid,
        levels=levels,
        extend="both",
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

def plot_pressure_regime_2d(
    pressure_functions,
    time_days,
    regime_name,
    figures_directory,
    nx=100,
    ny=50,
    pressure_min=1.0,
    pressure_max=2.0,
):
    """
    Plots pressure fields from different times of the same regime.

    Parameters
    ----------
    pressure_functions : list
        List of Firedrake pressure functions.

    time_days : list
        Simulation times in days.

    regime_name : str
        Name of the operating regime.

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

    pressure_grids = []

    for pressure_function in pressure_functions:
        pressure_grid = sample_field(
            pressure_function,
            XX,
            YY,
        )

        pressure_grids.append(pressure_grid)

    levels = np.linspace(
        pressure_min - 1.0e-3,
        pressure_max + 1.0e-3,
        21,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )

    axes = axes.ravel()

    contour = None

    for axis, pressure_grid, day in zip(
        axes,
        pressure_grids,
        time_days,
    ):
        contour = axis.contourf(
            XX,
            YY,
            pressure_grid,
            levels=levels,
            extend="both",
        )

        axis.set_xlabel("X")
        axis.set_ylabel("Y")

        axis.set_title(
            f"t = {day:.1f} dias"
        )

        axis.set_aspect("equal")

    colorbar = figure.colorbar(
        contour,
        ax=axes,
        location="right",
        shrink=0.9,
    )

    colorbar.set_label(
        "Pressão adimensional"
    )

    figure.suptitle(
        f"Campo de pressão — {regime_name}",
        fontsize=14,
    )

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / f"pressure_2d_{regime_name}.png"
    )

    figure.savefig(
        filename,
        dpi=300,
    )

    plt.close(figure)

def plot_mid_height_transient_pressure_profiles(
    pressure_functions,
    time_days,
    regime_names,
    figures_directory,
    y_mid=0.1,
    nx=200,
):
    """
    Plots mid-height pressure profiles in three separate subplots.

    Parameters
    ----------
    pressure_functions : list
        Stored Firedrake pressure functions.

    time_days : list
        Physical days associated with each pressure function.

    regime_names : list
        Operating regime associated with each profile.

    figures_directory : Path
        Directory where the figure will be saved.

    y_mid : float
        Fixed dimensionless height.

    nx : int
        Number of points along the X direction.
    """

    x_values = np.linspace(0.0, 1.0, nx)

    pressure_profiles = []

    for pressure_function in pressure_functions:

        pressure_values = np.array([
            point_value(
                pressure_function,
                x_value,
                y_mid,
            )
            for x_value in x_values
        ])

        pressure_profiles.append(pressure_values)

    regime_order = [
        "Injection",
        "Stop",
        "Production",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True,
    )

    for ax, regime_name in zip(axes, regime_order):

        for pressure_values, time_day, current_regime in zip(
            pressure_profiles,
            time_days,
            regime_names,
        ):

            if current_regime != regime_name:
                continue

            ax.plot(
                x_values,
                pressure_values,
                label=f"Day {time_day}",
            )

        ax.set_title(regime_name)
        ax.set_xlabel("Dimensionless position, X")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Dimensionless pressure, P")

    fig.suptitle(
        "Mid-height transient pressure profiles",
    )

    fig.tight_layout()

    filename = (
        figures_directory
        / "mid_height_transient_profiles.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def get_velocity_magnitude(velocity_function):
    """
    Calcula a magnitude da velocidade de Darcy adimensional.

    Parameters
    ----------
    velocity_function : Firedrake Function
        Campo vetorial de velocidade em DG0.

    Returns
    -------
    numpy.ndarray
        Magnitude da velocidade em cada elemento.
    """

    velocity_values = velocity_function.dat.data_ro

    velocity_magnitude = np.sqrt(
        velocity_values[:, 0]**2
        + velocity_values[:, 1]**2
    )

    return velocity_magnitude

def plot_velocity_2d(
    velocity_function,
    time_days,
    figures_directory,
):
    """
    Plota a magnitude da velocidade de Darcy adimensional em 2D.
    """

    # Magnitude da velocidade em cada elemento
    velocity_magnitude = get_velocity_magnitude(velocity_function)

    # Coordenadas dos vértices da malha
    mesh = velocity_function.function_space().mesh()
    coordinates = mesh.coordinates.dat.data_ro

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    cell_node_map = mesh.coordinates.cell_node_map()
    cells = cell_node_map.values

    fig, ax = plt.subplots(
        figsize=(8, 4),
        dpi=300,
    )

    # Plotagem dos valores por elemento
    collection = ax.tripcolor(
        x,
        y,
        cells,
        velocity_magnitude,
        shading="flat",
    )

    fig.colorbar(
        collection,
        ax=ax,
        label=r"$|\mathbf{U}|$",
    )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_title(
        f"Dimensionless Darcy velocity — t = {time_days} days"
    )

    ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig(
        figures_directory
        / f"velocity-2d-t{time_days}.png"
    )

    plt.close(fig)

def plot_velocity_regime_2d(
    velocity_functions,
    time_days,
    regime_name,
    figures_directory,
):
    """
    Plota a magnitude da velocidade de Darcy em diferentes
    tempos de um mesmo regime.
    """

    velocity_magnitudes = [
        get_velocity_magnitude(velocity_function)
        for velocity_function in velocity_functions
    ]

    velocity_min = min(
        np.min(values)
        for values in velocity_magnitudes
    )

    velocity_max = max(
        np.max(values)
        for values in velocity_magnitudes
    )

    mesh = velocity_functions[0].function_space().mesh()

    coordinates = mesh.coordinates.dat.data_ro

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    cells = mesh.coordinates.cell_node_map().values

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )

    axes = axes.ravel()

    levels = np.linspace(
        velocity_min,
        velocity_max,
        21,
    )

    contour = None

    for ax, velocity_magnitude, day in zip(
        axes,
        velocity_magnitudes,
        time_days,
    ):

        contour = ax.tripcolor(
            x,
            y,
            cells,
            velocity_magnitude,
            shading="flat",
            vmin=velocity_min,
            vmax=velocity_max,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_title(
            f"t = {day} dias"
        )

        ax.set_aspect("equal")

    colorbar = fig.colorbar(
        contour,
        ax=axes,
        location="right",
        shrink=0.9,
    )

    colorbar.set_label(
        r"$|\mathbf{U}|$"
    )

    fig.suptitle(
        f"Campo de velocidade de Darcy — {regime_name}",
        fontsize=14,
    )

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / f"velocity_2d_{regime_name}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)

def plot_mid_height_transient_velocity_profiles(
    velocity_functions,
    time_days,
    regime_names,
    figures_directory,
    y_mid=0.1,
    nx=200,
):
    """
    Plots mid-height Darcy velocity magnitude profiles
    in three separate subplots.
    """

    x_values = np.linspace(0.0, 1.0, nx)

    velocity_profiles = []

    for velocity_function in velocity_functions:

        velocity_values = np.array([
            velocity_function.at((x_value, y_mid))
            for x_value in x_values
        ])

        velocity_magnitudes = np.linalg.norm(
            velocity_values,
            axis=1,
        )

        velocity_profiles.append(velocity_magnitudes)

    regime_order = [
        "Injection",
        "Stop",
        "Production",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True,
    )

    for ax, regime_name in zip(axes, regime_order):

        for velocity_values, time_day, current_regime in zip(
            velocity_profiles,
            time_days,
            regime_names,
        ):

            if current_regime != regime_name:
                continue

            ax.plot(
                x_values,
                velocity_values,
                label=f"Day {time_day}",
            )

        ax.set_title(regime_name)
        ax.set_xlabel("Dimensionless position, X")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(
        r"Dimensionless Darcy velocity magnitude, $|\mathbf{U}|$"
    )

    fig.suptitle(
        "Mid-height transient Darcy velocity profiles",
    )

    fig.tight_layout()

    filename = (
        figures_directory
        / "mid_height_transient_velocity_profiles.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
