import numpy as np
import matplotlib.pyplot as plt
import os


def plot_viscosity_vs_pressure(model_function, T, model_name):
    """
    μ vs P (T fixa)
    """

    os.makedirs("figures/viscosity", exist_ok=True)

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

    filename = f"figures/viscosity/viscosity_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_viscosity_vs_temperature(model_function, P, model_name):
    """
    μ vs T (P fixa)
    """

    os.makedirs("figures/viscosity", exist_ok=True)

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

    filename = f"figures/viscosity/viscosity_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()

def plot_viscosity_vs_pressure_comparison(models, T):
    """
    Compara viscosidade vs pressão para múltiplos modelos (T fixa)
    """

    os.makedirs("figures/viscosity", exist_ok=True)

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

        plt.plot(
            pressures_MPa,
            viscosities,
            label=model_name,
            linewidth=2
        )

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Viscosidade (Pa·s)")

    plt.title(
        f"Comparação Viscosidade vs Pressão (T = {T} K)"
    )

    plt.legend()
    plt.grid()

    plt.savefig(
        "figures/viscosity/viscosity_vs_P_comparison.png",
        dpi=300
    )

    plt.close()


def plot_viscosity_vs_temperature_comparison(models, P):
    """
    Compara viscosidade vs temperatura para múltiplos modelos (P fixa)
    """

    os.makedirs("figures/viscosity", exist_ok=True)

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

        plt.plot(
            temperatures,
            viscosities,
            label=model_name,
            linewidth=2
        )

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Viscosidade (Pa·s)")

    plt.title(
        f"Comparação Viscosidade vs Temperatura (P = {P/1e6:.1f} MPa)"
    )

    plt.legend()
    plt.grid()

    plt.savefig(
        "figures/viscosity/viscosity_vs_T_comparison.png",
        dpi=300
    )
    plt.close()