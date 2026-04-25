# src/models/thermopack_model.py

from thermopack.cubic import PengRobinson
import numpy as np



# Inicializando modelo Peng-Robinson
eos = PengRobinson("H2")  # fluido (H2)
z_comp = [1.0]  # Composição: hidrogênio puro
PHASE_GAS = 1   # Fase: 1 = vapor

# Constantes
M_H2 = 2.016e-3  # kg/mol   # massa molar do hidrogênio
R = 8.314        # J/mol·K  # constante dos gases


# =========================
# FUNÇÕES PRINCIPAIS
# =========================

def calculate_Z(P, T):
    return eos.zfac(T, P, z_comp, phase=PHASE_GAS)


def molar_volume(P, T):
    return eos.specific_volume(T, P, z_comp, phase=PHASE_GAS)


# =========================
# DENSIDADE - DUAS FORMAS
# =========================

def density_via_volume(P, T):
    """
    rho = M / V_m     
    """
    Vm = molar_volume(P, T)
    return M_H2 / Vm


def density_via_Z(P, T):
    """
    rho = (P * M) / (Z * R * T)
    """
    Z = calculate_Z(P, T)
    return (P * M_H2) / (Z * R * T)


# =========================
# COMPARAÇÃO
# =========================

def density_difference(P, T):
    """
    Diferença absoluta entre os dois métodos
    """
    rho1 = density_via_volume(P, T)
    rho2 = density_via_Z(P, T)
    return abs(rho1 - rho2)


def density_relative_error(P, T):
    """
    Erro relativo (%)
    """
    rho1 = density_via_volume(P, T)
    rho2 = density_via_Z(P, T)
    return abs((rho1 - rho2) / rho1) * 100

