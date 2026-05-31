"""
run_properties.py

Gera gráficos de propriedades termodinâmicas do hidrogênio para comparação entre modelos.

Rodar com:

python -m src.runners.run_properties
"""
# =================

import matplotlib
matplotlib.use("Agg")  # para evitar o problema que deu

import matplotlib.pyplot as plt
# ================= 
# coolprop
from src.properties.coolprop_model import (
    calculate_Z as Z_cp,
    calculate_density as rho_cp_func,
    calculate_viscosity as mu_cp
)

# thermo
from src.properties.thermo_model import (
    calculate_Z as Z_th,
    calculate_density as rho_th_func,
    calculate_viscosity as mu_th
)

# thermo PR
from src.properties.thermo_pr_model import (
    calculate_Z as Z_pr,
    calculate_density as rho_pr_func
)

# thermopack
from src.properties.thermopack_model import (
    calculate_Z as Z_tp,
    density_via_Z as rho_tp_func,
    density_via_volume,
    density_relative_error,
    density_difference
)

# neqsim
from src.properties.neqsim_model import (
    calculate_Z as z_neq,
    calculate_density as rho_neq,
    calculate_viscosity as mu_neq
)

# plots
from src.plotting.properties.plot_z import (
    plot_Z_vs_pressure, 
    plot_Z_vs_temperature, 
    plot_Z_vs_pressure_comparison, 
    plot_Z_vs_temperature_comparison,)
from src.plotting.properties.plot_density import (
    plot_density_vs_pressure, 
    plot_density_vs_temperature, 
    plot_density_vs_Z,
    plot_density_vs_pressure_comparison,
    plot_density_vs_temperature_comparison,
    plot_density_vs_Z_comparison)
from src.plotting.properties.plot_viscosity import (
    plot_viscosity_vs_pressure, 
    plot_viscosity_vs_temperature,
    plot_viscosity_vs_pressure_comparison,
    plot_viscosity_vs_temperature_comparison )




# parâmetros
T = 300  # temperatura em K
P = 1e7  # 100 bar = 10 MPa = 1e7 Pa 

# ===================
# MODELOS ORGANIZADOS
Z_models = {
    "CoolProp": Z_cp,
    "Thermo": Z_th,
    "Thermo_PR": Z_pr,
    "Thermopack": Z_tp,
    "NeqSim": z_neq
}

density_models = {
    "CoolProp": rho_cp_func,
    "Thermo": rho_th_func,
    "Thermo_PR": rho_pr_func,
    "Thermopack": rho_tp_func,
    "NeqSim": rho_neq
}

viscosity_models = {
    "CoolProp": mu_cp,
    "Thermo": mu_th,
    "NeqSim": mu_neq
}

# ===================

print("=== Comparação de densidade ===")

for name, func in density_models.items():
    try:
        value = func(P, T)
    except Exception:
        value = float("nan")
    print(f"{name}: {value}")

# ========
# plots:
# ========

for name, func in Z_models.items():
    plot_Z_vs_pressure(func, T, f"Hydrogen - {name}")
    plot_Z_vs_temperature(func, P, f"Hydrogen - {name}")

for name, func in density_models.items():
    plot_density_vs_pressure(func, T, f"Hydrogen - {name}")
    plot_density_vs_temperature(func, P, f"Hydrogen - {name}")

# densidade vs Z (precisa de Z + rho consistentes)
for name in Z_models.keys():
    plot_density_vs_Z(
        density_models[name],
        Z_models[name],
        T,
        f"Hydrogen - {name}"
    )
 
for name, func in viscosity_models.items():
    plot_viscosity_vs_pressure(func, T, f"Hydrogen - {name}")
    plot_viscosity_vs_temperature(func, P, f"Hydrogen - {name}")


# =========================
# PLOTS COMPARAÇÕES
plot_Z_vs_pressure_comparison(Z_models, T)
plot_Z_vs_temperature_comparison(Z_models, P)
plot_density_vs_pressure_comparison(density_models, T)
plot_density_vs_temperature_comparison(density_models, P)
plot_density_vs_Z_comparison(density_models, Z_models, T)
plot_viscosity_vs_pressure_comparison(viscosity_models, T)
plot_viscosity_vs_temperature_comparison(viscosity_models, P)

"""
rho_v = density_via_volume(P, T)
rho_z = rho_tp_func(P, T)

print("=== Thermopack Consistency Check ===")
print("Densidade via volume:", rho_v)
print("Densidade via Z:", rho_z)
print("Diferença absoluta:", density_difference(P, T))
print("Erro relativo (%):", density_relative_error(P, T))
"""