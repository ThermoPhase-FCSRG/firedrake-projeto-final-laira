# src/models/thermo_model.py

from thermo import Chemical


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

c = Chemical('hydrogen', T=300, P=20e6)

print("Vm:", c.Vmg)
print("Vm ideal:", c.Vmg_ideal)
print("Método:", c.VolumeGas.method)
print("Métodos:", c.VolumeGas.all_methods)
print("Método de pressão:", c.VolumeGas.method_P)
print("Z:", c.Z)