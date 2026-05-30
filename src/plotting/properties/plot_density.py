import numpy as np
import matplotlib.pyplot as plt
import os


def plot_density_vs_pressure(model_function, T, model_name):
    """
    Gráfico densidade vs pressão (T fixa)
    """

    os.makedirs("figures/density", exist_ok=True)

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

    filename = f"figures/density/density_vs_P_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()

def plot_density_vs_temperature(model_function, P, model_name):
    """
    Gráfico densidade vs temperatura (P fixa)
    """

    os.makedirs("figures/density", exist_ok=True)

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

    filename = f"figures/density/density_vs_T_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()    


def plot_density_vs_Z(model_density, model_Z, T, model_name):
    """
    Gráfico densidade vs Z (variando pressão)
    """

    os.makedirs("figures/density", exist_ok=True)

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

    filename = f"figures/density/density_vs_Z_{model_name}.png"
    plt.savefig(filename, dpi=300)

    plt.close()    



def plot_density_vs_pressure_comparison(models, T):
    """
    Compara densidade vs Pressão para múltiplos modelos (T fixa)
    """
    os.makedirs("figures/density", exist_ok=True)

    pressures = np.linspace(1e5, 2e7, 100)
    pressures_MPa = pressures / 1e6

    plt.figure()

    for model_name, model_function in models.items():
        densities = []

        for P in pressures:
            rho = model_function(P, T)
            densities.append(rho)

        plt.plot(pressures_MPa, densities, label=model_name)

    plt.xlabel("Pressão (MPa)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Pressão (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig("figures/density/density_vs_P_comparison.png", dpi=300)
    plt.close()

def plot_density_vs_temperature_comparison(models, P):
    """
    Compara densidade vs Temperatura para múltiplos modelos (P fixa)
    """
    os.makedirs("figures/density", exist_ok=True)

    temperatures = np.linspace(200, 500, 100)

    plt.figure()

    for model_name, model_function in models.items():
        densities = []

        for T in temperatures:
            rho = model_function(P, T)
            densities.append(rho)

        plt.plot(temperatures, densities, label=model_name)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Temperatura (P = {P/1e6:.1f} MPa)")

    plt.legend()
    plt.grid()

    plt.savefig("figures/density/density_vs_T_comparison.png", dpi=300)
    plt.close()



def plot_density_vs_Z_comparison(density_models, Z_models, T):
    """
    Compara densidade vs Z para múltiplos modelos (T fixa)
    """

    os.makedirs("figures/density", exist_ok=True)

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

        plt.plot(Z_values, densities, label=model_name)

    plt.xlabel("Fator de compressibilidade (Z)")
    plt.ylabel("Densidade (kg/m³)")
    plt.title(f"Comparação Densidade vs Z (T = {T} K)")

    plt.legend()
    plt.grid()

    plt.savefig(
        "figures/density/density_vs_Z_comparison.png",
        dpi=300
    )
    plt.close()