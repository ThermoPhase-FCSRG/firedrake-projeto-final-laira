import numpy as np
import matplotlib.pyplot as plt
from src.utils.paths import FIGURES_PROP_VISCOSITY

LINESTYLES = {
    "CoolProp": "-",
    "Thermo": "-",
    "Thermo_PR": (0, (8, 2)),
    "Thermopack": (0, (2, 2)),
    "NeqSim": "-.",
    "Torch-Flash": "--",
    "Torch-Flash_PR": (0, (8, 2, 2, 2))
}

MARKERS = {
    "CoolProp": "o",
    "Thermo": None,
    "Thermo_PR": "D",
    "Thermopack": "v",
    "NeqSim": "^",
    "Torch-Flash": "s",
    "Torch-Flash_PR": "X"
}

MARKEVERY = {
    "CoolProp": (0, 12),
    "Thermo": None,
    "Thermo_PR": (2, 12),
    "Thermopack": (5, 12),
    "NeqSim": (3, 12),
    "Torch-Flash": (7, 12),
    "Torch-Flash_PR": (9, 12)
}

def plot_viscosity_vs_pressure(model_function, T, model_name):
    """
    μ vs P (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 50)
    pressures_MPa = pressures / 1e6
    viscosities = []

    for P in pressures:
        mu = model_function(P, T)
        viscosities.append(mu)

    plt.figure()
    plt.plot(pressures_MPa, viscosities)

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Viscosidade (Pa·s)")
    plt.title(f"Viscosidade vs Pressão - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_VISCOSITY / f"viscosity_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_viscosity_vs_temperature(model_function, P, model_name):
    """
    μ vs T (P fixa)
    """

    temperatures = np.linspace(200, 500, 50)
    viscosities = []

    for T in temperatures:
        mu = model_function(P, T)
        viscosities.append(mu)

    plt.figure()
    plt.plot(temperatures, viscosities)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Viscosidade (Pa·s)")
    plt.title(f"Viscosidade vs Temperatura - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_VISCOSITY / f"viscosity_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()

def plot_viscosity_vs_pressure_comparison(models, T, labels=None):
    """
    Compara viscosidade vs pressão para múltiplos modelos (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 100)
    pressures_MPa = pressures / 1e6

    plt.figure()

    for model_name, model_function in models.items():

        viscosities = []

        for P in pressures:
            try:
                mu = model_function(P, T)
            except Exception:
                mu = np.nan

            viscosities.append(mu)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            pressures_MPa,
            viscosities,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Viscosidade (Pa·s)")
    plt.title(f"Comparação Viscosidade vs Pressão (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig(
        FIGURES_PROP_VISCOSITY / "viscosity_vs_P_comparison.png",
        dpi=300
    )

    plt.close()


def plot_viscosity_vs_temperature_comparison(models, P, labels=None):
    """
    Compara viscosidade vs temperatura para múltiplos modelos (P fixa)
    """

    temperatures = np.linspace(200, 500, 100)

    plt.figure()

    for model_name, model_function in models.items():

        viscosities = []

        for T in temperatures:
            try:
                mu = model_function(P, T)
            except Exception:
                mu = np.nan

            viscosities.append(mu)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            temperatures,
            viscosities,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Viscosidade (Pa·s)")
    plt.title(f"Comparação Viscosidade vs Temperatura (P = {P/1e6:.1f} MPa)")

    plt.legend()
    plt.grid()

    plt.savefig(
        FIGURES_PROP_VISCOSITY / "viscosity_vs_T_comparison.png",
        dpi=300
    )

    plt.close()