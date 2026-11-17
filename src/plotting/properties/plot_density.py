import numpy as np
import matplotlib.pyplot as plt
from src.utils.paths import FIGURES_PROP_DENSITY

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

def plot_density_vs_pressure(model_function, T, model_name):
    """
    Gráfico densidade vs pressão (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 50)
    pressures_MPa = pressures / 1e6
    densities = []

    for P in pressures:
        rho = model_function(P, T)
        densities.append(rho)

    plt.figure()
    plt.plot(pressures_MPa, densities)
    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Densidade vs Pressão - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_DENSITY / f"density_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()

def plot_density_vs_temperature(model_function, P, model_name):
    """
    Gráfico densidade vs temperatura (P fixa)
    """

    temperatures = np.linspace(200, 500, 50)
    densities = []

    for T in temperatures:
        rho = model_function(P, T)
        densities.append(rho)

    plt.figure()
    plt.plot(temperatures, densities)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Densidade vs Temperatura - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_DENSITY / f"density_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()    

def plot_density_vs_Z(model_density, model_Z, T, model_name):
    """
    Gráfico densidade vs Z (variando pressão)
    """

    pressures = np.linspace(1e5, 2e7, 100)

    densities = []
    Z_values = []

    for P in pressures:
        rho = model_density(P, T)
        Z = model_Z(P, T)

        densities.append(rho)
        Z_values.append(Z)
        # print(P, Z, rho)

    plt.figure()
    plt.plot(Z_values, densities)

    plt.xlabel("Fator de compressibilidade (Z)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Densidade vs Z - {model_name}")

    plt.grid()

    filename = FIGURES_PROP_DENSITY / f"density_vs_Z_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()    



def plot_density_vs_pressure_comparison(models, T, labels=None):
    """
    Compara densidade vs Pressão para múltiplos modelos (T fixa)
    """
    pressures = np.linspace(1e5, 2e7, 100)
    pressures_MPa = pressures / 1e6

    plt.figure()

    for model_name, model_function in models.items():
        densities = []

        for P in pressures:
            rho = model_function(P, T)
            densities.append(rho)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            pressures_MPa,
            densities,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Pressão (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig(FIGURES_PROP_DENSITY / "density_vs_P_comparison.png", dpi=300)
    plt.close()

def plot_density_vs_temperature_comparison(models, P, labels=None):
    """
    Compara densidade vs Temperatura para múltiplos modelos (P fixa)
    """
    temperatures = np.linspace(200, 500, 100)

    plt.figure()

    for model_name, model_function in models.items():
        densities = []

        for T in temperatures:
            rho = model_function(P, T)
            densities.append(rho)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            temperatures,
            densities,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Temperatura (P = {P/1e6:.1f} MPa)")

    plt.legend()
    plt.grid()

    plt.savefig(FIGURES_PROP_DENSITY / "density_vs_T_comparison.png", dpi=300)
    plt.close()



def plot_density_vs_Z_comparison(density_models, Z_models, T, labels=None):
    """
    Compara densidade vs Z para múltiplos modelos (T fixa)
    """

    pressures = np.linspace(1e5, 2e7, 100)

    plt.figure()

    for model_name in density_models.keys():

        densities = []
        Z_values = []

        density_function = density_models[model_name]
        Z_function = Z_models[model_name]

        for P in pressures:
            rho = density_function(P, T)
            Z = Z_function(P, T)

            densities.append(rho)
            Z_values.append(Z)

        legend_label = labels.get(model_name, model_name) if labels else model_name

        plt.plot(
            Z_values,
            densities,
            label=legend_label,
            linestyle=LINESTYLES.get(model_name, "-"),
            marker=MARKERS.get(model_name),
            markevery=MARKEVERY.get(model_name),
            markersize=5,
            markerfacecolor="white",
            linewidth=1.8
        )

    plt.xlabel("Fator de compressibilidade (Z)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Z (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig(
        FIGURES_PROP_DENSITY / "density_vs_Z_comparison.png",
        dpi=300
    )
    plt.close()