import numpy as np
import matplotlib.pyplot as plt
from src.utils.paths import FIGURES_PROP_Z


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

def plot_Z_vs_pressure(model_function, T, model_name):
    """
    Gráfico Z vs Pressão (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 50)
    pressures_mpa = pressures / 1e6
    Z_values = []

    for P in pressures:
        Z = model_function(P, T)
        Z_values.append(Z)

    plt.figure()
    plt.plot(pressures_mpa, Z_values)
    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Z")
    plt.title(f"Z vs Pressão - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_Z / f"Z_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_Z_vs_temperature(model_function, P, model_name):
    """
    Gráfico Z vs Temperatura (P fixa)
    """

    temperatures = np.linspace(200, 500, 50)
    Z_values = []

    for T in temperatures:
        Z = model_function(P, T)
        Z_values.append(Z)

    plt.figure()
    plt.plot(temperatures, Z_values)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Z")
    plt.title(f"Z vs Temperatura - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_Z / f"Z_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_Z_vs_pressure_comparison(models, T, labels=None):
    """
    Compara Z vs Pressão para múltiplos modelos (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 100)
    pressures_mpa = pressures / 1e6

    plt.figure()

    for model_name, model_function in models.items():
        Z_values = []

        for P in pressures:
            Z = model_function(P, T)
            Z_values.append(Z)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            pressures_mpa,
            Z_values,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Z")
    plt.title(f"Comparação Z vs Pressão (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig(FIGURES_PROP_Z / "Z_vs_P_comparison.png", dpi=300)
    plt.close()

def plot_Z_vs_temperature_comparison(models, P, labels=None):
    """
    Compara Z vs Temperatura para múltiplos modelos (P fixa)
    """

    temperatures = np.linspace(200, 500, 100)

    plt.figure()

    for model_name, model_function in models.items():
        Z_values = []

        for T in temperatures:
            Z = model_function(P, T)
            Z_values.append(Z)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            temperatures,
            Z_values,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Z")
    plt.title(f"Comparação Z vs Temperatura (P = {P/1e6:.1f} MPa)")

    plt.legend()
    plt.grid()

    plt.savefig(FIGURES_PROP_Z / "Z_vs_T_comparison.png", dpi=300)
    plt.close()