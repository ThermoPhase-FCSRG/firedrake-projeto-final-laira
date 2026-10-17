import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    phase_properties,
    state_derivatives,
    two_phase_flash,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    temperature=runtime.tensor(270.0),       # K
    pressure=runtime.tensor(3.0e6),          # Pa
    composition=runtime.tensor([0.5, 0.5]),  # mole fractions
)

# Evaluate the supplied homogeneous state without solving an equilibrium.
properties = phase_properties(model, state, phase="stable")
print(properties.compressibility_factor)
print(properties.fugacities, properties.log_fugacities)
print(properties.chemical_potentials)
print(properties.molar_helmholtz_energy, properties.molar_gibbs_energy)
print(properties.reduced_helmholtz_energy, properties.reduced_gibbs_energy)

# Derivatives are evaluated with PyTorch automatic differentiation.
derivatives = state_derivatives(model, state)
print(derivatives.dfugacity_dpressure)
print(derivatives.dchemical_potential_dtemperature)

# Stability-tested two-phase TP flash.
equilibrium = two_phase_flash(model, state)
print(equilibrium.phase_fractions)
print(equilibrium.phase_kinds, equilibrium.phase_regime)
for phase in equilibrium.phases:
    print(phase.composition)