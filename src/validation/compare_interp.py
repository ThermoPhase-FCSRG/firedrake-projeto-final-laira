# Script para comparar os resultados da interpolação com os valores reais calculados por CoolProp""

import numpy as np
import matplotlib.pyplot as plt

from src.properties import coolprop  #,  thermo, thermopack
from src.interpolation.interpolators import load_interpolators


# 🔹 1. gerar pontos de teste
def generate_test_points(n=100):
    P_test = np.random.uniform(1e5, 5e7, n)
    T_test = np.random.uniform(250, 500, n)
    return P_test, T_test


# 🔹 2. calcular erro
def compute_error(P_test, T_test, Z_interp, Z_real):

    errors = []

    for P, T in zip(P_test, T_test):
        Z_r = Z_real(P, T)
        Z_i = Z_interp([[P, T]])[0]

        error = abs((Z_i - Z_r) / Z_r)
        errors.append(error)

    return np.array(errors)


# 🔹 3. plot
def plot_histogram(errors, name):

    plt.figure()
    plt.hist(errors, bins=30)
    plt.xlabel("Erro relativo")
    plt.ylabel("Frequência")
    plt.title(f"Erro da interpolação - {name}")
    plt.grid()
    # plt.savefig(f"histogram_{name}.png")


# 🔹 4. função principal de validação
def run_validation(property_module, name):
    
    Z_real = property_module.Z

    # IMPORTANTE: carregar interpolador da biblioteca correta
    Z_interp, _, _ = load_interpolators(name.lower())

    P_test, T_test = generate_test_points(200)

    errors = compute_error(P_test, T_test, Z_interp, Z_real)

    print(f"\n--- {name} ---")
    print("Erro médio:", errors.mean())
    print("Erro máximo:", errors.max())

    plot_histogram(errors, name)


# 🔹 5. execução
# if __name__ == "__main__":
    # run_validation(coolprop, "CoolProp")
    # run_validation(thermo, "Thermo")
    # run_validation(thermopack, "ThermoPack")