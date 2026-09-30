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
