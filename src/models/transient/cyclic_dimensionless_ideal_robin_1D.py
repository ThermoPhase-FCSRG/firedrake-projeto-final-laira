"""
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (1D)
Formulação variacional com Firedrake

Descrição do problema: (transiente)
----------------------
Resolve-se o problema transiente de escoamento monofásico de um gás ideal
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada é:

    φ ∂p/∂t = (k/μ) ∂/∂x ( p ∂p/∂x )

Condições de contorno:
    Condição de Robin na fronteira do poço injetor e produtor:
        U · n = Gamma (P - P_w)

Condição inicial:
    p(x, 0) = p_r,  ∀ x ∈ Ω

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG),
e a discretização temporal é feita via esquema implícito de Euler.
"""

# Importing libraries
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import FIGURES_SIM_IDEAL_TRANSIENT_ROBIN
from src.plotting.simulations.plot_1d import (
    plot_pressure_1d,
    plot_pressure_regime_1d,
    plot_velocity_1d,
    plot_velocity_regime_1d,
    plot_pressure_comparison_1d,
    plot_velocity_comparison_1d,
)

# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # comprimento caracteristico 
X_left, X_right = 0.0, 1  # valor adimensionalizado
mesh = IntervalMesh(numel, X_left, X_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Initial condition
# ic = Constant(1e7)
ic = Constant(1.0)  # valor adimensionalizado

# Trial and Test functions
p = Function(V)
p_k = Function(V)
v = TestFunction(V)

p_injection_left = Constant(2.0)
p_production_right = Constant(1.0)


# Physical parameters
phi = Constant(0.15)        # porosity
# kappa = Constant(2.6647e-13)     # permeability [m^2]
# kappa = Constant(1.0e-18)
kappa = Constant(1.0e-16)   # =0.0101325 mD TESTE (!!!)
mu = Constant(0.94e-5)         # viscosity [Pa.s]  in a temperature of 50C
f = Constant(0.0)            # source term  

# Characteristic scales
p_c = 1.0e7                   # characteristic pressure [Pa]; o mesmo que pr

t_c = float(mu) * float(phi) * L**2 / (float(kappa) * p_c)  # Characteristic time

print("Characteristic time =", t_c, "s")
print("Characteristic time =", t_c / (24*3600), "days")

# ------------------
# Time parameters
numero_de_dias = 360
t_total = numero_de_dias * 24 * 3600  # physical total time =  360 days
T_total = t_total / t_c  # adimensional total time
dT = (t_total / numero_de_dias) / t_c  # adimensional time step

def operation_mode(T):
    time_days = T * t_c / (24 * 3600)
    cycle_length = 90.0
    cycle_day = time_days % cycle_length

    if cycle_day < 30:
        return "injection"
    elif cycle_day < 60:
        return "stop"
    else:
        return "production"


for day in [0, 10, 30, 40, 60, 70, 90, 100, 120, 130, 150, 160, 180, 200, 300, 350]:
    T_test = day * 24 * 3600 / t_c
    print(day, operation_mode(T_test))


# ----------------
# Parâmetro adimensional da condição de Robin
Gamma_open = Constant(1000.0)

Gamma_left = Constant(0.0)
Gamma_right = Constant(0.0)


# -----------------
def update_well_pressures(T):
    mode = operation_mode(T)

    if mode == "injection":
        Gamma_left.assign(Gamma_open)
        Gamma_right.assign(0.0)

    elif mode == "stop":
        Gamma_left.assign(0.0)
        Gamma_right.assign(0.0)

    elif mode == "production":
        Gamma_left.assign(0.0)
        Gamma_right.assign(Gamma_open)


# ------------------

# Assigning the IC
p_k.assign(ic)
p.assign(ic)

""" não vai precisar para o caso admensionalizado
# Compressibility factor fitted from PR-EoS in terms of pressure
def Z(p):
    return 1.0

# Non-linear pressure term
def fp(p):
    return p / Z(p)
""" 
# Residual variational formulation
F = inner((p - p_k) / dT, v) * dx + inner(p * grad(p), grad(v)) * dx
F -= f * v * dx
F += Gamma_left * p * (p - p_injection_left) * v * ds(1)   # Robin condition at x=0
F += Gamma_right * p * (p - p_production_right) * v * ds(2)  # Robin condition at x=L

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'pc_type': 'lu'
}

# Iterating and solving over the time
T = dT
step = 0
x_values = mesh.coordinates.dat.data_ro # Laira

sol_values = []
p_values_deg1 = []
p_dimensional_values = []
psol_deg1 = Function(Vref)

# ===== Espaço para velocidade de Darcy =====
V_u = FunctionSpace(mesh, "DG", 0)   # espaço descontínuo por elemento
u = Function(V_u, name="Darcy velocity")

# Coordenada do centro de cada elemento (para plot step)
x = SpatialCoordinate(mesh)
x_cell = Function(V_u)
x_cell.project(x[0])
x_cells = x_cell.dat.data_ro.copy()

u_time_series = [] # Lista para guardar velocidade ao longo do tempo
u_dimensional_time_series = [] # Lista para guardar velocidade dimensional ao longo do tempo

u_c = float(kappa) * p_c / (float(mu) * L)
print("Characteristic velocity =", u_c, "m/s")


while T <= T_total:
    step += 1

    update_well_pressures(T)
    solve(F == 0, p, solver_parameters=solver_parameters)
    

    print(
    f"day={step:3d}, mode={operation_mode(T)}, "
    f"P_left={p.dat.data_ro[0]:.6f}, "
    f"P_right={p.dat.data_ro[-1]:.6f}"
)


    # ===== Pós-processamento da velocidade (TRANSIENTE) =====
    # u_expr = -(kappa / mu) * p.dx(0)    # OBSERVAÇÃO: NÃO ESTÁ ADMENSIONALIZADA AINDA
    u_expr =  -p.dx(0)  # adimensionalized Darcy velocity
    u.project(u_expr)
    

    # -----
    u_vals = u.dat.data_ro.copy()  # representa U adimensional 
    u_time_series.append(u_vals)

    u_dimensional_vals = u_vals * u_c      # u = U * u_c
    u_dimensional_time_series.append(u_dimensional_vals)

    sol_vec = p.dat.data_ro.copy()   # adimensionalized pressure (P)
    sol_values.append(sol_vec)

    p_dimensional_vec = sol_vec * p_c  # dimensionalized pressure   (p = P *pc)
    p_dimensional_values.append(p_dimensional_vec)

    psol_deg1.project(p)
    p_vec_deg1 = psol_deg1.dat.data_ro.copy()
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    T += dT

print("Number of stored solutions:", len(p_values_deg1))

# ==================================
# *** Plotting ***

plot_pressure_1d(
    x_values=x_values,
    pressure_values=p_values_deg1,
    time_steps=[1, 5, 10, 20, 50, 100, 200, 300, 350],
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)

# ============================================================
# Pressão separada por regime

pressure_regime_days = [
    1, 10, 20, 29,
    30, 40, 50, 59,
    60, 70, 80, 89,
]

pressure_regime_names = [
    "injection", "injection", "injection", "injection",
    "stop", "stop", "stop", "stop",
    "production", "production", "production", "production",
]

pressure_regime_values = [
    p_values_deg1[day - 1]
    for day in pressure_regime_days
]

plot_pressure_regime_1d(
    pressure_values=pressure_regime_values,
    x_values=x_values,
    time_days=pressure_regime_days,
    regime_names=pressure_regime_names,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)
# ============================================================
# Velocidade separada por regime

plot_velocity_regime_1d(
    x_values=x_cells,
    velocity_values=u_time_series,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)

# =======================================================
# velocidade 
plot_velocity_1d(
    x_values=x_cells,
    velocity_values=u_time_series,
    time_steps=[1, 5, 10, 20, 50, 100, 200, 300, 350],
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)

# ============================================
# Comparação adimensional x dimensional

steps_to_compare = [1, 50, 100]

plot_pressure_comparison_1d(
    x_values=x_values,
    pressure_values=sol_values,
    pressure_dimensional_values=p_dimensional_values,
    p_c=p_c,
    time_steps=steps_to_compare,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)

plot_velocity_comparison_1d(
    x_values=x_cells,
    velocity_values=u_time_series,
    velocity_dimensional_values=u_dimensional_time_series,
    u_c=u_c,
    time_steps=steps_to_compare,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN,
)
