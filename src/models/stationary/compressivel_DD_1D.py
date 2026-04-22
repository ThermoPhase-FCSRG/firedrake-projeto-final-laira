"""
Escoamento monofásico compressível de gás hidrogenio em meio poroso (1D)
Formulação variacional com Firedrake

Descrição do problema: (estacionário)
----------------------
Resolve-se o problema estacionário de escoamento monofásico de um gás 
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é compressivel,
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada é:

    (k/μ) ∂/∂x ( p ∂p/∂x ) = 0

onde:
    p   = pressão
    φ   = porosidade
    k   = permeabilidade
    μ   = viscosidade do fluido

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG), 

rodar com: python -m src.models.stationary.compressivel_DD_1D
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import FIGURES_SIM

# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # alterado de 50 para 200m para ver melhor a evolução da pressão
mesh = IntervalMesh(numel, 0.0, L)

# Function space
degree = 1
V = FunctionSpace(mesh, "CG", degree)

# Boundary conditions
p_left = Constant(2.0e7)   # 200 bar
p_right = Constant(1.0e7)  # 100 bar

bc_left = DirichletBC(V, p_left, 1)
bc_right = DirichletBC(V, p_right, 2)

bcs = [bc_left, bc_right]

# Unknown and test function
p = Function(V, name="Pressure")
v = TestFunction(V)

# Initial guess
p.assign(1.5e7)  # Define um valor inicial para o método de Newton começar a iteração.

# Physical parameters
# kappa = Constant(1.0e-18)
kappa = Constant(1.0e-16)   # =0.0101325 mD TESTE (!!!)
mu = Constant(0.94e-5)

# Compressibility (gás ideal)
def Z(p):
    return 1.0

def fp(p):
    return p / Z(p)

# Variational formulation (STEADY STATE)
F = (kappa / mu) * inner(fp(p) * grad(p), grad(v)) * dx

# Solver parameters
solver_parameters = {
    "mat_type": "aij",  # matriz esparsa padrão
    "snes_type": "newtonls",   # Método de Newton com linha de busca
    "pc_type": "lu"  # resolve o sistema linear com fatoração LU
}

# Solve
solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)


# Post-processing 
V_u = FunctionSpace(mesh, "DG", 0) # Function space for velocity
u = Function(V_u, name="Darcy velocity")
u_expr = -(kappa / mu) * p.dx(0)
u.project(u_expr)

u_values = u.dat.data_ro.copy() # Pega os valores da velocidade de Darcy
x = SpatialCoordinate(mesh)
x_cell = Function(V_u)
x_cell.project(x[0])

x_cells = x_cell.dat.data_ro.copy()


# """
# Plot da velocidade numérica
plt.figure(dpi=300, figsize=(8, 6))
plt.step(
    x_cells,
    u_values,
    where="mid",
    linewidth=2,
    label="Velocidade Darcy (FEM)"
)

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$u$ [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-velocity.png")

# =========================
# Velocidade analítica
# =========================

# Parâmetros
pw = float(p_left)
pr = float(p_right)

a = (pr**2 - pw**2) / L

# Pressão analítica nos centros das células
p_analytical_cells = np.sqrt(pw**2 + a * x_cells)

# Velocidade analítica
u_analytical = -(float(kappa) / float(mu)) * a / (2.0 * p_analytical_cells)


# solution
x_values = mesh.coordinates.dat.data_ro # Pega os valores das coordenadas dos nós da malha.
p_values = p.dat.data_ro / 1e3  # Pega os valores da pressão numérica e converte de Pa para kPa.

# Analytical solution (steady state)
pw = float(p_left)
pr = float(p_right)

p_analytical = np.sqrt(
    pw**2 + (pr**2 - pw**2) * x_values / L
)

p_analytical = p_analytical / 1e3  # kPa

# =========================
# Plotting
# =========================

plt.figure(dpi=300, figsize=(8, 6))

plt.step(
    x_cells,
    u_values,
    where="mid",
    linewidth=2,
    label="Velocidade Darcy (FEM)"
)

plt.plot(
    x_cells,
    u_analytical,
    "--",
    linewidth=2,
    label="Velocidade Darcy (Analítica)"
)

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$u$ [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-velocity-comparison.png")

# """
plt.figure(dpi=300, figsize=(8, 6))
plt.plot(x_values, p_values, label="Steady state")
plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [kPa]")
plt.xlim(x_values.min(), x_values.max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-pressure.png")
# plt.show()
# """

plt.figure(dpi=300, figsize=(8, 6))

plt.plot(
    x_values,
    p_values,
    "o",
    markersize=3,
    label="FEM (CG1)"
)

plt.plot(
    x_values,
    p_analytical,
    "-",
    linewidth=2,
    label="Solução analítica"
)

plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [kPa]")
plt.xlim(x_values.min(), x_values.max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-pressure-comparison.png")
# plt.show()

