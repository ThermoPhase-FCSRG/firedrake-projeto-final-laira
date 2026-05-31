"""
Descrição do problema: (transiente)
----------------------
Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- efeitos gravitacionais são desprezados.

A equação governante considerada é:
    φ ∂p/∂t = (k/μ) ∂/∂x ( p ∂p/∂x )

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)
Condição inicial:
    p(x, 0) = p_r,  ∀ x ∈ Ω
"""

# Importing libraries
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from src.interpolation.interpolators import load_interpolators
from src.utils.paths import FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DD

# ================================
# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # alterado de 50 para 200m para ver melhor a evolução da pressão
x_left, x_right = 0.0, L
mesh = IntervalMesh(numel, x_left, x_right)

# ================================

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# ================================

# Boundary condition (Dirichlet) and Initial condition
boundary_value_left = 2e7
bc_left = DirichletBC(V, boundary_value_left, 1)  # Boundary condition in 1 marked bounds (left)
boundary_value_right = 1e7
bc_right = DirichletBC(V, boundary_value_right, 2)
bcs = [bc_left, bc_right]

ic = Constant(1e7)

# ================================

# Trial and Test functions
p = Function(V)
p_k = Function(V)
v = TestFunction(V)

# ================================

# Physical parameters
phi = Constant(0.15)        # porosity
kappa = Constant(1.0e-16)   # permeability [m^2]    =0.0101325 mD TESTE (!!!)
# mu = Constant(0.94e-5)         # viscosity [Pa.s]  in a temperature of 50C
mu_field = Function(V, name="Viscosity")
mu_field.assign(0.94e-5)
f = Constant(0.0)            # source term  
T_const = 300.0

z_field = Function(V, name="Compressibility")
z_field.assign(1.0)
Z_interp, rho_interp, mu_interp = load_interpolators()

# ===============================

# Time parameters
# T_total = 4.147e7  # 480 days
# dt = T_total / 500.

# T_total = 2 * 24 * 3600   # 2 dias em segundos
# dt = T_total / 200        # passos menores para ver a evolução

# T_total = 30 * 24 * 3600  # 30 dias
# dt = T_total / 200

T_total = 180 * 24 * 3600  # 120 dias
dt = T_total / 200

# ===============================

# Assigning the IC
p_k.assign(ic)
p.assign(ic)

# ===================================

# Non-linear pressure term
def fp(p):
    return p / z_field

# ===================================

# Residual variational formulation
F = phi * inner((fp(p) - fp(p_k)) / dt, v) * dx + (kappa / mu_field) * inner(fp(p) * grad(p), grad(v)) * dx
F -= f * v * dx

# ===================================

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'pc_type': 'lu'
}

# ================================

# Picard parameters
max_picard = 20
tol_picard = 1e-6

# ===================================

# Iterating and solving over the time
t = dt
step = 0
# diego: x_values = mesh.coordinates.vector().dat.data
x_values = mesh.coordinates.dat.data_ro # Laira

"""
sol_values = []
p_values_deg1 = []
""" 


psol_deg1 = Function(Vref)
steps_to_plot = [1, 5, 10, 20, 50, 100, 200]
pressure_snapshots = {}
velocity_snapshots = {}

# ===============================
# Espaço para velocidade de Darcy
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

    print(f"\n=== Time step {step} ===")

    # -------------------------
    # Picard loop
    # -------------------------
    for picard in range(max_picard):

        p_old = p.copy(deepcopy=True)

        solve(
            F == 0,
            p,
            bcs=bcs,
            solver_parameters=solver_parameters
        )

        p_vals = p.dat.data_ro

        # μ(P,T)
        mu_vals = np.array([
            mu_interp([val, T_const])[0]
            for val in p_vals
        ])

        # Z(P,T)
        z_vals = np.array([
            Z_interp([val, T_const])[0]
            for val in p_vals
        ])

        mu_field.dat.data[:] = mu_vals
        z_field.dat.data[:] = z_vals

        error = np.linalg.norm(
            p.dat.data - p_old.dat.data
        )

        print(
            f"Picard {picard+1}: "
            f"error = {error:.3e}"
        )

        print(
            f"mu = [{mu_vals.min():.3e}, "
            f"{mu_vals.max():.3e}]"
        )

        print(
            f"Z = [{z_vals.min():.3e}, "
            f"{z_vals.max():.3e}]"
        )

        if error < tol_picard:
            break

    # -------------------------
    # Darcy velocity
    u_expr = -(kappa / mu_field) * p.dx(0)
    u.project(u_expr)

    # -------------------------
    # Save selected snapshots
    if step in steps_to_plot:

        psol_deg1.project(p)

        pressure_snapshots[step] = (
            psol_deg1.dat.data_ro.copy()
        )

        velocity_snapshots[step] = (
            u.dat.data_ro.copy()
        )

    # -------------------------
    # advance in time
    # -------------------------
    p_k.assign(p)

    t += dt

# ===============================

# Plotting

# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
# steps_to_plot = [1, 10, 30, 60, 120, 360, 480]
steps_to_plot = [1, 5, 10, 20, 50, 100, 200]  

for step in steps_to_plot:
    ax.plot(
        x_values,
        pressure_snapshots[step] / 1e3,
        label=f"Step {step}"
    )

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
plt.savefig(FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DD / "transient-DD-pressure.png")
    

# ========================================

# plotting velocity profiles over time
plt.figure(dpi=300, figsize=(8, 6))

for step in steps_to_plot:
    plt.step(
        x_cells,
        velocity_snapshots[step],
        where="mid",
        linewidth=2,
        label=f"Step {step}"
    )

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"Darcy velocity [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DD / "transient-DD-velocity.png")
