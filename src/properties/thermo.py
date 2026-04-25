# src/models/thermo_model.py

from thermo import Chemical

# usando Peng-Robinson 

# Criar objeto do fluido
fluid = Chemical('hydrogen')

def calculate_density(P, T):
    """
    Densidade usando thermo
    """
    fluid.T = T
    fluid.P = P
    return fluid.rho

def calculate_Z(P, T):
    """
    Fator de compressibilidade
    """
    fluid.T = T
    fluid.P = P
    return fluid.Z

def calculate_viscosity(P, T):
    """
    Viscosidade
    """
    fluid.T = T
    fluid.P = P
    return fluid.mu