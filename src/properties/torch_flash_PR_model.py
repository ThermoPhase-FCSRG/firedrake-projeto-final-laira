"""Propriedades termodinâmicas do hidrogênio usando torch-flash.

Z: equação de estado de Peng-Robinson (1976).
Viscosidade: modelo de estados correspondentes de Pedersen.

Entradas:
    P: pressão em Pa
    T: temperatura em K

Saídas:
    Z: fator de compressibilidade
    densidade: kg/m³
    viscosidade: Pa·s
"""

import torch
from torch_flash import (
    component_set,
    peng_robinson_1976,
    corresponding_states_viscosity,
)

DTYPE = torch.float64

# Componente puro: hidrogênio.
COMPONENTS = component_set(("hydrogen",))
COMPOSITION = torch.tensor([1.0], dtype=DTYPE)

# Equação de estado de Peng-Robinson (1976).
EOS = peng_robinson_1976(COMPONENTS)

MOLAR_MASS = COMPONENTS.molar_mass[0]

R_UNIVERSAL = 8.314462618  # J/(mol·K)


def calculate_Z(P, T):
    """Calcula o fator de compressibilidade do hidrogênio."""
    pressure = torch.as_tensor(P, dtype=DTYPE)
    temperature = torch.as_tensor(T, dtype=DTYPE)

    molar_volume = EOS.molar_volume(temperature, pressure, COMPOSITION)
    # Z = EOS.compressibility_factor(temperature, molar_volume)
    Z = pressure * molar_volume / (R_UNIVERSAL * temperature)

    return Z.item()


def calculate_density(P, T):
    """Calcula a densidade do hidrogênio em kg/m³."""
    pressure = torch.as_tensor(P, dtype=DTYPE)
    temperature = torch.as_tensor(T, dtype=DTYPE)

    molar_volume = EOS.molar_volume(temperature, pressure, COMPOSITION)
    density = MOLAR_MASS / molar_volume

    return density.item()


def calculate_viscosity(P, T):
    """Calcula a viscosidade dinâmica do hidrogênio em Pa·s."""
    pressure = torch.as_tensor(P, dtype=DTYPE)
    temperature = torch.as_tensor(T, dtype=DTYPE)

    viscosity = corresponding_states_viscosity(
        temperature=temperature,
        pressure=pressure,
        composition=COMPOSITION,
        components=COMPONENTS,
        phase="vapor",
    )

    return viscosity.item()

if __name__ == "__main__":
    P = 2e7  # Pa
    T = 300.0  # K

    print("Peng-Robinson (1976)")
    print(f"Z: {calculate_Z(P, T):.6f}")
    print(f"Densidade: {calculate_density(P, T):.6f} kg/m³")
    print(f"Viscosidade: {calculate_viscosity(P, T):.8e} Pa·s")