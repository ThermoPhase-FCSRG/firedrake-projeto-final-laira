from thermo import ChemicalConstantsPackage
from thermo import PRMIX
from thermo import CEOSGas

fluids = ['hydrogen']

constants, correlations = ChemicalConstantsPackage.from_IDs(fluids)

eos_kwargs = {
    "Tcs": constants.Tcs,
    "Pcs": constants.Pcs,
    "omegas": constants.omegas
}


def calculate_Z(P, T):
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, zs=[1.0], T=T, P=P)
    return float(gas.Z())


def calculate_density(P, T):
    R = 4124  # J/kg.K

    Z = calculate_Z(P, T)

    rho = P / (Z * R * T)

    return rho