"""
Validação da interpolação de propriedades termodinâmicas

Compara:
- valores reais (CoolProp)
- valores interpolados

Saídas:
- erro médio
- erro máximo
- histogramas
- erro vs pressão
"""

import numpy as np
import matplotlib.pyplot as plt

from src.properties import coolprop
from src.interpolation.interpolators import load_interpolators
from src.utils.paths import FIGURES_VAL


# =========================
# 1. gerar pontos de teste
# =========================
def generate_test_points(n=200):
    P_test = np.random.uniform(1e7, 2e7, n)
    T_test = np.random.uniform(200, 500, n)
    return P_test, T_test


# =========================
# 2. calcular erro
# =========================
def compute_error(P_test, T_test, Z_interp):

    Z_real_vals = []
    Z_interp_vals = []

    for P, T in zip(P_test, T_test):

        Z_r = coolprop.Z(P, T)
        Z_i = Z_interp([P, T])[0]

        Z_real_vals.append(Z_r)
        Z_interp_vals.append(Z_i)

    Z_real_vals = np.array(Z_real_vals)
    Z_interp_vals = np.array(Z_interp_vals)

    error = np.abs((Z_interp_vals - Z_real_vals) / Z_real_vals)

    return error, Z_real_vals, Z_interp_vals


# =========================
# 3. plots
# =========================

def plot_histogram(errors):
    plt.figure(dpi=300)
    plt.hist(errors, bins=30)
    plt.xlabel("Erro relativo")
    plt.ylabel("Frequência")
    plt.title("Erro da interpolação (Z)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_VAL / "hist_error_Z.png")


def plot_error_vs_pressure(P_test, errors):
    plt.figure(dpi=300)
    plt.scatter(P_test, errors, s=10)
    plt.xlabel("Pressão [Pa]")
    plt.ylabel("Erro relativo")
    plt.title("Erro vs Pressão")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_VAL / "error_vs_pressure.png")


def plot_error_vs_temperature(T_test, errors):
    plt.figure(dpi=300)
    plt.scatter(T_test, errors, s=10)
    plt.xlabel("Temperatura [K]")
    plt.ylabel("Erro relativo")
    plt.title("Erro vs Temperatura")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_VAL / "error_vs_temperature.png")


# =========================
# 4. execução principal
# =========================
def run_validation():

    print("\n=== Validação da interpolação (CoolProp) ===")

    # carregar interpoladores
    Z_interp, _, _ = load_interpolators()

    # gerar pontos
    P_test, T_test = generate_test_points()

    # calcular erro
    errors, Z_real_vals, Z_interp_vals = compute_error(
        P_test, T_test, Z_interp
    )

    # métricas
    print(f"Erro médio:  {errors.mean():.3e}")
    print(f"Erro máximo: {errors.max():.3e}")

    # plots
    plot_histogram(errors)
    plot_error_vs_pressure(P_test, errors)
    plot_error_vs_temperature(T_test, errors)


# =========================
# 5. rodar
# =========================
if __name__ == "__main__":
    run_validation()