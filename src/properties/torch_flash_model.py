# torch_flash_model.py
"""Propriedades termodinâmicas do hidrogênio usando torch-flash.

Z: equação de estado de Leachman.
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
    leachman_normal_hydrogen,
    corresponding_states_viscosity,
)


DTYPE = torch.float64

# Equação de estado de Leachman para hidrogênio normal.
EOS = leachman_normal_hydrogen(dtype=DTYPE)

# Componente puro usado no cálculo da viscosidade.
COMPONENTS = component_set(("hydrogen",))
COMPOSITION = torch.tensor([1.0], dtype=DTYPE)

MOLAR_MASS = COMPONENTS.molar_mass[0]


def calculate_Z(P, T):
    """Calcula o fator de compressibilidade do hidrogênio."""
    pressure = torch.as_tensor(P, dtype=DTYPE)
    temperature = torch.as_tensor(T, dtype=DTYPE)

    molar_volume = EOS.molar_volume(temperature, pressure)
    Z = EOS.compressibility_factor(temperature, molar_volume)

    return Z.item()


def calculate_density(P, T):
    """Calcula a densidade do hidrogênio em kg/m³."""
    pressure = torch.as_tensor(P, dtype=DTYPE)
    temperature = torch.as_tensor(T, dtype=DTYPE)

    molar_volume = EOS.molar_volume(temperature, pressure)
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
