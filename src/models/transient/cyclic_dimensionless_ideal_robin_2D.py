"""
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (2D)
Formulação variacional com Firedrake

Admite-se que:
- problema transiente (dependente do tempo),
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante adimensionalizada considerada é:

    ∂P/∂T = ∂/∂X ( p ∂p/∂X ) + ∂/∂Y ( p ∂p/∂Y )

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

from src.utils.paths import FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D
from src.plotting.simulations.plot_2d import (
    plot_pressure_2d,
    plot_pressure_regime_2d,
    plot_mid_height_transient_profiles,
)

# Mesh definition
numel_x = 100 # mudei de 200 para 100 (!!!)
numel_y = 10 # rever depois (!!!)

L = 500.0   # comprimento característico  

# domínio adimensionalizado 
X_left, X_right = 0.0, 1.0 
Y_bottom, Y_top = 0.0, 0.2

mesh = RectangleMesh(numel_x, numel_y, X_right - X_left, Y_top - Y_bottom)

# ==========================================
# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# ==========================================
# Initial condition
""" isso significa P(X,Y,0) = 1.0, ou seja, 
todo o reservatório começa com pressão adimensional igual a 1."""
ic = Constant(1.0)  # valor adimensionalizado

# ==========================================
# Trial and Test functions
p = Function(V)
p_k = Function(V)
v = TestFunction(V)

p_injection_left = Constant(2.0)
p_production_right = Constant(1.0)

# ==========================================
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


# ==================================================
# Parâmetro adimensional da condição de Robin
Gamma_open = Constant(1000.0)

Gamma_left = Constant(0.0)
Gamma_right = Constant(0.0)

# ==================================================
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


# ==================================================
# Assigning the IC
p_k.assign(ic)
p.assign(ic)

# ==================================================
""" não vai precisar para o caso admensionalizado
# Compressibility factor fitted from PR-EoS in terms of pressure
def Z(p):
    return 1.0

# Non-linear pressure term
def fp(p):
    return p / Z(p)
""" 

# ==================================================
# Residual variational formulation
F = (inner((p - p_k) / dT, v) * dx          
     + inner(p * grad(p), grad(v)) * dx)
F -= f * v * dx
F += Gamma_left * p * (p - p_injection_left) * v * ds(1)   # Robin condition at x=0
F += Gamma_right * p * (p - p_production_right) * v * ds(2)  # Robin condition at x=L

# ==================================================
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
p_snapshots = []

# ==================================================
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
    
    """
    print(
    f"day={step:3d}, mode={operation_mode(T)}, "
    f"P_left={p.dat.data_ro[0]:.6f}, "
    f"P_right={p.dat.data_ro[-1]:.6f}" 
    )
    """

    
    # ===== Pós-processamento da velocidade (TRANSIENTE) =====
    # u_expr = -(kappa / mu) * p.dx(0)    # OBSERVAÇÃO: NÃO ESTÁ ADMENSIONALIZADA AINDA
    u_expr =  -p.dx(0)  # adimensionalized Darcy velocity
    u.project(u_expr)

    u_vals = u.dat.data_ro.copy()  # representa U adimensional 
    u_time_series.append(u_vals)

    u_dimensional_vals = u_vals * u_c      # u = U * u_c
    u_dimensional_time_series.append(u_dimensional_vals)

    sol_vec = p.dat.data_ro.copy()   # adimensionalized pressure (P)
    sol_values.append(sol_vec)

    p_snapshot = Function(V)
    p_snapshot.assign(p)

    p_snapshots.append(p_snapshot)

    p_dimensional_vec = sol_vec * p_c  # dimensionalized pressure   (p = P *pc)
    p_dimensional_values.append(p_dimensional_vec)

    psol_deg1.project(p)
    p_vec_deg1 = psol_deg1.dat.data_ro.copy()
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    T += dT

# print("Number of stored solutions:", len(p_values_deg1))
# print("Número de snapshots:", len(p_snapshots))

def get_regime_snapshots(p_snapshots, days):
    """
    Selects pressure snapshots for specified physical days.
    """

    pressure_functions = [
        p_snapshots[day - 1]
        for day in days
    ]

    return pressure_functions, days

# =================================
# Plotting
# =================================
plot_pressure_2d(
    pressure_function=p_snapshots[0],
    time_days=1.0,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Regime de injeção
injection_functions, injection_days = get_regime_snapshots(
    p_snapshots,
    days=[1, 10, 20, 29],
)

plot_pressure_regime_2d(
    pressure_functions=injection_functions,
    time_days=injection_days,
    regime_name="injection_cycle_1",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Segundo ciclo de injeção
injection_functions, injection_days = get_regime_snapshots(
    p_snapshots,
    days=[90, 100, 110, 119],
)

plot_pressure_regime_2d(
    pressure_functions=injection_functions,
    time_days=injection_days,
    regime_name="injection_cycle_2",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Terceiro ciclo de injeção
injection_functions, injection_days = get_regime_snapshots(
    p_snapshots,
    days=[180, 190, 200, 209],
)
plot_pressure_regime_2d(
    pressure_functions=injection_functions,
    time_days=injection_days,
    regime_name="injection_cycle_3",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Quarto ciclo de injeção
injection_functions, injection_days = get_regime_snapshots(
    p_snapshots,
    days=[270, 280, 290, 299],
)
plot_pressure_regime_2d(
    pressure_functions=injection_functions,
    time_days=injection_days,
    regime_name="injection_cycle_4",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# ----------------------------------------------------------------
# Regime de poços fechados
# Stop - ciclo 1
stop_functions, stop_days = get_regime_snapshots(
    p_snapshots,
    days=[30, 40, 50, 59],
)

plot_pressure_regime_2d(
    pressure_functions=stop_functions,
    time_days=stop_days,
    regime_name="stop_cycle_1",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Stop - ciclo 2
stop_functions, stop_days = get_regime_snapshots(
    p_snapshots,
    days=[120, 130, 140, 149],
)

plot_pressure_regime_2d(
    pressure_functions=stop_functions,
    time_days=stop_days,
    regime_name="stop_cycle_2",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Stop - ciclo 3
stop_functions, stop_days = get_regime_snapshots(
    p_snapshots,
    days=[210, 220, 230, 239],
)

plot_pressure_regime_2d(
    pressure_functions=stop_functions,
    time_days=stop_days,
    regime_name="stop_cycle_3",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Stop - ciclo 4
stop_functions, stop_days = get_regime_snapshots(
    p_snapshots,
    days=[300, 310, 320, 329],
)

plot_pressure_regime_2d(
    pressure_functions=stop_functions,
    time_days=stop_days,
    regime_name="stop_cycle_4",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# ----------------------------------------------------------------
# Regime de produção
# Production - ciclo 1
production_functions, production_days = get_regime_snapshots(
    p_snapshots,
    days=[60, 70, 80, 89],
)

plot_pressure_regime_2d(
    pressure_functions=production_functions,
    time_days=production_days,
    regime_name="production_cycle_1",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Production - ciclo 2
production_functions, production_days = get_regime_snapshots(
    p_snapshots,
    days=[150, 160, 170, 179],
)

plot_pressure_regime_2d(
    pressure_functions=production_functions,
    time_days=production_days,
    regime_name="production_cycle_2",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Production - ciclo 3
production_functions, production_days = get_regime_snapshots(
    p_snapshots,
    days=[240, 250, 260, 269],
)

plot_pressure_regime_2d(
    pressure_functions=production_functions,
    time_days=production_days,
    regime_name="production_cycle_3",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# Production - ciclo 4
production_functions, production_days = get_regime_snapshots(
    p_snapshots,
    days=[330, 340, 350, 359],
)

plot_pressure_regime_2d(
    pressure_functions=production_functions,
    time_days=production_days,
    regime_name="production_cycle_4",
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# -------------------------------------------------------------------------
# Mid-height transient profiles
# ---------------------------------------------------------

mid_height_functions = [
    # Injection
    p_snapshots[0],    # Day 1
    p_snapshots[14],   # Day 15
    p_snapshots[28],   # Day 29

    # Stop
    p_snapshots[29],   # Day 30
    p_snapshots[44],   # Day 45
    p_snapshots[58],   # Day 59

    # Production
    p_snapshots[59],   # Day 60
    p_snapshots[74],   # Day 75
    p_snapshots[88],   # Day 89
]

mid_height_days = [
    1, 15, 29,
    30, 45, 59,
    60, 75, 89,
]

mid_height_regimes = [
    "Injection", "Injection", "Injection",
    "Stop", "Stop", "Stop",
    "Production", "Production", "Production",
]

plot_mid_height_transient_profiles(
    pressure_functions=mid_height_functions,
    time_days=mid_height_days,
    regime_names=mid_height_regimes,
    figures_directory=FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D,
)

# ---------------------------------------------------------












"""
# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
steps_to_plot = [1, 5, 10, 20, 50, 100, 200, 300, 350]  
# steps_to_plot = [1, 10, 20, 29, 30,40, 50, 59, 60,70, 80, 89, 90]

# X_values = x_values / L
for i in steps_to_plot:
    time_days = i
    ax.plot(x_values, p_values_deg1[i-1], label=f"t = {time_days} days")

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$X$ ')
plt.ylabel(r'Pressure')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

# Displaying the plot
plt.tight_layout()
plt.savefig(FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D / "cyclic-dimensionless-ideal-transient-robin-pressure.png")
    
#plt.show()

# ============================================================
# Comparação: 
steps_to_compare = [1, 50, 100]


# pressão adimensional x pressão dimensional 
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

for i in steps_to_compare:

    # Pressão adimensional
    ax.plot(
        x_values,
        sol_values[i-1],
        linewidth=2,
        label=f"Adimensional: t = {i} dias"
    )

    # Pressão dimensional convertida novamente para adimensional
    ax.plot(
        x_values,
        p_dimensional_values[i-1] / p_c,
        linestyle="--",
        linewidth=2,
        label=f"Dimensional: t = {i} dias"
    )

ax.set_xlabel(r"$X$")
ax.set_ylabel(r"Pressure / $p_c$")
ax.set_xlim(0, 1)

ax.grid(True)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D
    / "comparison-dimensionless-dimensional-pressure-cyclic-transient-robin.png"
)
# ----------------
# velocidade adimensional x velocidade dimensional

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

for i in steps_to_compare:

    # Velocidade adimensional
    ax.plot(
        x_cells,
        u_time_series[i-1],
        linewidth=2,
        label=f"Adimensional: t = {i} dias"
    )

    # Velocidade dimensional convertida novamente para adimensional
    ax.plot(
        x_cells,
        u_dimensional_time_series[i-1] / u_c,
        linestyle="--",
        linewidth=2,
        label=f"Dimensional: t = {i} dias"
    )

ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$U = u/u_c$")
ax.set_xlim(0, 1)

ax.grid(True)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D
    / "comparison-dimensionless-dimensional-velocity-cyclic-transient-robin.png"
)

# =====================================================
# plotting velocity profiles over time
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

for i in steps_to_plot:
    ax.step(
        x_cells,
        u_time_series[i-1],
        where="mid",
        linewidth=2,
        label=f"t = {i} days"
    )

ax.set_xlabel(r"$X$")
ax.set_ylabel(r"Dimensionless Darcy velocity, $U$")
ax.set_xlim(0, 1)
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES_SIM_IDEAL_TRANSIENT_ROBIN_2D
    / "cyclic-dimensionless-ideal-transient-robin-velocity.png"
)
"""