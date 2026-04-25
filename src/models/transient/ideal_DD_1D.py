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

from src.utils.paths import FIGURES_SIM

# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # alterado de 50 para 200m para ver melhor a evolução da pressão
x_left, x_right = 0.0, L
mesh = IntervalMesh(numel, x_left, x_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Boundary condition (Dirichlet) and Initial condition
boundary_value_left = 2e7
bc_left = DirichletBC(V, boundary_value_left, 1)  # Boundary condition in 1 marked bounds (left)
boundary_value_right = 1e7
bc_right = DirichletBC(V, boundary_value_right, 2)
bcs = [bc_left, bc_right]


ic = Constant(1e7)


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

# ------------------
# Time parameters
# T_total = 4.147e7  # 480 days
# dt = T_total / 500.

# T_total = 2 * 24 * 3600   # 2 dias em segundos
# dt = T_total / 200        # passos menores para ver a evolução

# T_total = 30 * 24 * 3600  # 30 dias
# dt = T_total / 200

T_total = 180 * 24 * 3600  # 120 dias
dt = T_total / 200
# ------------------

# Assigning the IC
p_k.assign(ic)
p.assign(ic)

# Compressibility factor fitted from PR-EoS in terms of pressure
def Z(p):
    return 1.0

# Non-linear pressure term
def fp(p):
    return p / Z(p)

# Residual variational formulation
F = phi * inner((fp(p) - fp(p_k)) / dt, v) * dx + (kappa / mu) * inner(fp(p) * grad(p), grad(v)) * dx
F -= f * v * dx

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'pc_type': 'lu'
}

# Iterating and solving over the time
t = dt
step = 0
# diego: x_values = mesh.coordinates.vector().dat.data
x_values = mesh.coordinates.dat.data_ro # Laira

sol_values = []
p_values_deg1 = []
psol_deg1 = Function(Vref)

# ===== Espaço para velocidade de Darcy =====
V_u = FunctionSpace(mesh, "DG", 0)   # espaço descontínuo por elemento
u = Function(V_u, name="Darcy velocity")

# Coordenada do centro de cada elemento (para plot step)
x = SpatialCoordinate(mesh)
x_cell = Function(V_u)
x_cell.project(x[0])
x_cells = x_cell.dat.data_ro.copy()

# Lista para guardar velocidade ao longo do tempo
u_time_series = []

while t <= T_total:
    step += 1
    # print('============================')
    # print('\ttime =', t)
    # print('\tstep =', step)
    # print('============================')

    solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)
    # diego : sol_vec = np.array(p.vector().dat.data)
    # sol_values.append(sol_vec)
    # psol_deg1.project(p)
    # p_vec_deg1 = np.array(psol_deg1.vector().dat.data)
    # p_values_deg1.append(p_vec_deg1)

    # ===== Pós-processamento da velocidade (TRANSIENTE) =====
    u_expr = -(kappa / mu) * p.dx(0)
    u.project(u_expr)

    u_vals = u.dat.data_ro.copy()
    u_time_series.append(u_vals)

    sol_vec = p.dat.data_ro.copy()
    sol_values.append(sol_vec)

    psol_deg1.project(p)
    p_vec_deg1 = psol_deg1.dat.data_ro.copy()
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    t += dt

# *** Plotting ***

# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
# steps_to_plot = [1, 10, 30, 60, 120, 360, 480]
steps_to_plot = [1, 5, 10, 20, 50, 100, 200]  


for i in steps_to_plot:
    ax.plot(x_values, p_values_deg1[i-1] / 1e3, label=('Time step %i' % (i)))

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'Pressure [kPa]')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

# Displaying the plot
plt.tight_layout()
plt.savefig(FIGURES_SIM / "ideal-transient-DD-pressure.png")
    
#plt.show()


# plotting velocity profiles over time
plt.figure(dpi=300, figsize=(8, 6))

for i in steps_to_plot:
    plt.step(
        x_cells,
        u_time_series[i-1],
        where="mid",
        linewidth=2,
        label=f"Day {i}"
    )

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"Darcy velocity [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "ideal-transient-DD-velocity.png")
