"""
Functions for plotting 1D simulation results.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_pressure_1d(
    x_values,
    pressure_values,
    time_steps,
    figures_directory,
):
    """
    Plots dimensionless pressure profiles at different times.
    """

    fig, ax = plt.subplots(
        dpi=300,
        figsize=(8, 6),
    )

    for day in time_steps:
        ax.plot(
            x_values,
            pressure_values[day - 1],
            label=f"t = {day} days",
        )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"Dimensionless pressure, $P$")
    ax.set_xlim(x_values.min(), x_values.max())

    ax.minorticks_on()
    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / "cyclic-dimensionless-ideal-transient-robin-pressure-1d.png"
    )

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)

def plot_pressure_regime_1d(
    pressure_values,
    x_values,
    time_days,
    regime_names,
    figures_directory,
):
    """
    Plots 1D pressure profiles separated by operating regime.
    """

    regime_order = [
        "injection",
        "stop",
        "production",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True,
    )

    for ax, regime_name in zip(axes, regime_order):

        for pressure, day, current_regime in zip(
            pressure_values,
            time_days,
            regime_names,
        ):

            if current_regime != regime_name:
                continue

            ax.plot(
                x_values,
                pressure,
                linewidth=2,
                label=f"Day {day}",
            )

        ax.set_title(regime_name.capitalize())
        ax.set_xlabel(r"$X$")
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(r"Dimensionless pressure, $P$")

    fig.suptitle(
        "Dimensionless pressure profiles by operating regime"
    )

    fig.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        figures_directory
        / "cyclic-dimensionless-ideal-transient-robin-pressure-regimes-1d.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_velocity_1d(
    x_values,
    velocity_values,
    time_steps,
    figures_directory,
):
    """
    Plots dimensionless Darcy velocity profiles at different times.
    """

    fig, ax = plt.subplots(
        dpi=300,
        figsize=(8, 6),
    )

    for day in time_steps:
        ax.plot(
            x_values,
            velocity_values[day - 1],
            label=f"t = {day} days",
        )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"Dimensionless Darcy velocity, $U$")
    ax.set_xlim(x_values.min(), x_values.max())

    ax.minorticks_on()
    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / "cyclic-dimensionless-ideal-transient-robin-velocity-1d.png"
    )

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)
        
def plot_velocity_regime_1d(
    x_values,
    velocity_values,
    figures_directory,
):
    regimes = {
        "Injection": [1, 10, 20, 29],
        "Stop": [30, 40, 50, 59],
        "Production": [60, 70, 80, 89],
    }

    fig, axes = plt.subplots(
        1,
        3,
        dpi=300,
        figsize=(15, 5),
        sharey=True,
    )

    for ax, (regime, days) in zip(axes, regimes.items()):

        for day in days:
            ax.plot(
                x_values,
                velocity_values[day - 1],
                linewidth=2,
                label=f"Day {day}",
            )

        ax.set_title(regime)
        ax.set_xlabel(r"$X$")
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(r"Dimensionless Darcy velocity, $U$")

    fig.suptitle(
        "Dimensionless Darcy velocity profiles by operating regime"
    )

    fig.tight_layout()

    fig.savefig(
        figures_directory
        / "cyclic-dimensionless-ideal-transient-robin-velocity-by-regime-1d.png"
    )

    plt.close(fig)

def plot_pressure_comparison_1d(
    x_values,
    pressure_values,
    pressure_dimensional_values,
    p_c,
    time_steps,
    figures_directory,
):
    """
    Compares dimensionless pressure with dimensional pressure
    converted back to dimensionless form.
    """

    fig, ax = plt.subplots(
        dpi=300,
        figsize=(8, 6),
    )

    for day in time_steps:

        # Dimensionless pressure
        ax.plot(
            x_values,
            pressure_values[day - 1],
            linewidth=2,
            label=f"Dimensionless: t = {day} days",
        )

        # Dimensional pressure converted to dimensionless
        ax.plot(
            x_values,
            pressure_dimensional_values[day - 1] / p_c,
            linestyle="--",
            linewidth=2,
            label=f"Dimensional: t = {day} days",
        )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"Pressure / $p_c$")
    ax.set_xlim(x_values.min(), x_values.max())

    ax.minorticks_on()
    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / "comparison-dimensionless-dimensional-pressure-cyclic-transient-robin.png"
    )

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)

def plot_velocity_comparison_1d(
    x_values,
    velocity_values,
    velocity_dimensional_values,
    u_c,
    time_steps,
    figures_directory,
):
    """
    Compares dimensionless Darcy velocity with dimensional velocity
    converted back to dimensionless form.
    """

    fig, ax = plt.subplots(
        dpi=300,
        figsize=(8, 6),
    )

    for day in time_steps:

        # Dimensionless velocity
        ax.plot(
            x_values,
            velocity_values[day - 1],
            linewidth=2,
            label=f"Dimensionless: t = {day} days",
        )

        # Dimensional velocity converted to dimensionless
        ax.plot(
            x_values,
            velocity_dimensional_values[day - 1] / u_c,
            linestyle="--",
            linewidth=2,
            label=f"Dimensional: t = {day} days",
        )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$U = u/u_c$")
    ax.set_xlim(x_values.min(), x_values.max())

    ax.minorticks_on()
    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    figures_directory = Path(figures_directory)
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        figures_directory
        / "comparison-dimensionless-dimensional-velocity-cyclic-transient-robin.png"
    )

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)

