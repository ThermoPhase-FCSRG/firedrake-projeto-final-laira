import numpy as np
import matplotlib.pyplot as plt
import os


def plot_Z_vs_pressure(model_function, T, model_name):
    """
    Gráfico Z vs Pressão (T fixa)
    """
    os.makedirs("figures/Z", exist_ok=True)

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

    filename = f"figures/Z/Z_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_Z_vs_temperature(model_function, P, model_name):
    """
    Gráfico Z vs Temperatura (P fixa)
    """
    os.makedirs("figures/Z", exist_ok=True)

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

    filename = f"figures/Z/Z_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()


def plot_Z_vs_pressure_comparison(models, T):
    """
    Compara Z vs Pressão para múltiplos modelos (T fixa)
    """
    os.makedirs("figures/Z", exist_ok=True)

    pressures = np.linspace(1e5, 2e7, 100)
    pressures_mpa = pressures / 1e6


    plt.figure()

    for model_name, model_function in models.items():
        Z_values = []

        for P in pressures:
            Z = model_function(P, T)
            Z_values.append(Z)

        plt.plot(pressures_mpa, Z_values, label=model_name)

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Z")
    plt.title(f"Comparação Z vs Pressão (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig("figures/Z/Z_vs_P_comparison.png", dpi=300)
    plt.close()


def plot_Z_vs_temperature_comparison(models, P):
    """
    Compara Z vs Temperatura para múltiplos modelos (P fixa)
    """
    os.makedirs("figures/Z", exist_ok=True)

    temperatures = np.linspace(200, 500, 100)

    plt.figure()

    for model_name, model_function in models.items():
        Z_values = []

        for T in temperatures:
            Z = model_function(P, T)
            Z_values.append(Z)

        plt.plot(temperatures, Z_values, label=model_name)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Z")
    plt.title(f"Comparação Z vs Temperatura (P = {P/1e6} MPa)")
    plt.legend()
    plt.grid()
    plt.savefig("figures/Z/Z_vs_T_comparison.png", dpi=300)
    plt.close()