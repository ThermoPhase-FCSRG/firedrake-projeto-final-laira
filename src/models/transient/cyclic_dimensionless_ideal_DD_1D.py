"""
Trabalho Final — Disciplina: GA 033 - Elementos Finitos
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

onde:
    p   = pressão
    φ   = porosidade
    k   = permeabilidade
    μ   = viscosidade do fluido

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)

Condição inicial:
    p(x, 0) = p_r,  ∀ x ∈ Ω

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG),
e a discretização temporal é feita via esquema implícito de Euler.
"""

"""
TESTANDO COM 2 COND DE CONTORNO DIRICHLET NO PROBLEMA TRANSIENTE
"""

# Importing libraries
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import FIGURES_SIM_IDEAL_TRANSIENT_DD

# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # comprimento caracteristico 
X_left, X_right = 0.0, 1  # valor adimensionalizado
mesh = IntervalMesh(numel, X_left, X_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Boundary condition (Dirichlet) and Initial condition
# boundary_value_left = 2e7
boundary_value_left = 2.0  # valor adimensionalizado
bc_left = DirichletBC(V, boundary_value_left, 1)  # Boundary condition in 1 marked bounds (left)
# boundary_value_right = 1e7
boundary_value_right = 1.0  # valor adimensionalizado
bc_right = DirichletBC(V, boundary_value_right, 2)
# bcs = [bc_left, bc_right]

# ic = Constant(1e7)
ic = Constant(1.0)  # valor adimensionalizado

# Trial and Test functions
p = Function(V)
p_k = Function(V)
v = TestFunction(V)

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
# t_total = 4.147e7  # 480 days
# dt = t_total / 500.

# t_total = 2 * 24 * 3600   # 2 dias em segundos
# dt = t_total / 200        # passos menores para ver a evolução

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


def hydraulic_bcs(T):
    mode = operation_mode(T)

    # Injection at x=0, cyclic operation
    if mode == "injection":
        return [
            DirichletBC(V, boundary_value_left, 1)
        ]

    elif mode == "stop":
        return []

    elif mode == "production":
        return [
            DirichletBC(V, boundary_value_right, 2)
        ]


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
p_dimentional_values = []
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


while T <= T_total:
    step += 1
    mode = operation_mode(T)
    bcs = hydraulic_bcs(T)
    solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)
    
    # ===== Pós-processamento da velocidade (TRANSIENTE) =====
    # u_expr = -(kappa / mu) * p.dx(0)    # OBSERVAÇÃO: NÃO ESTÁ ADMENSIONALIZADA AINDA
    u_expr =  -p.dx(0)  # adimensionalized Darcy velocity
    u.project(u_expr)

    u_vals = u.dat.data_ro.copy()
    u_time_series.append(u_vals)

    sol_vec = p.dat.data_ro.copy()   # adimensionalized pressure (P)
    sol_values.append(sol_vec)

    p_dimentional_vec = sol_vec * p_c  # dimensionalized pressure   (p = P *pc)
    p_dimentional_values.append(p_dimentional_vec)

    psol_deg1.project(p)
    p_vec_deg1 = psol_deg1.dat.data_ro.copy()
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    T += dT

print("Number of stored solutions:", len(p_values_deg1))

# *** Plotting ***

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
plt.savefig(FIGURES_SIM_IDEAL_TRANSIENT_DD / "cyclic-dimensionless-ideal-transient-DD-pressure.png")
    
#plt.show()

# ============================================================
# Comparação: pressão adimensional x pressão dimensional
# ============================================================

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

steps_to_compare = [1, 50, 100]
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
        p_dimentional_values[i-1] / p_c,
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
    FIGURES_SIM_IDEAL_TRANSIENT_DD
    / "comparison-dimensionless-dimensional-pressure-cyclic-transient-DD.png"
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
    FIGURES_SIM_IDEAL_TRANSIENT_DD
    / "cyclic-dimensionless-ideal-transient-DD-velocity.png"
)