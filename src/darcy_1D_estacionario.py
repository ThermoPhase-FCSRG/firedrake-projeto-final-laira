"""
Trabalho Final — Disciplina: GA 033 - Elementos Finitos
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (1D)
Formulação variacional com Firedrake

Descrição do problema: (estacionário)
----------------------
Resolve-se o problema estacionário de escoamento monofásico de um gás ideal
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
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

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG), 
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # alterado de 50 para 200m para ver melhor a evolução da pressão
mesh = IntervalMesh(numel, 0.0, L)

# Function space
degree = 1
V = FunctionSpace(mesh, "CG", degree)

# Boundary conditions
p_left = Constant(2.0e7)   # 200 bar
bc_left = DirichletBC(V, p_left, 1)
bcs = [bc_left]

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

# =========================
# Post-processing 
# =========================
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
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig("../figures/DN/k=1e-16/estacionario/velocity.png")
# """

# =========================
# Velocidade analítica
# =========================

# Parâmetros
pw = float(p_left)

# solution
x_values = mesh.coordinates.dat.data_ro
p_analytical = np.ones_like(x_values) * pw / 1e6  # MPa

u_analytical = np.zeros_like(x_cells) 

p_values = p.dat.data_ro / 1e6  # MPa


# =========================
# Plot velocidade
# =========================
plt.figure(dpi=300, figsize=(8, 6))

plt.step(x_cells, u_values, where="mid", linewidth=2, label="Velocidade Darcy (FEM)")
plt.plot(x_cells, u_analytical, "--", linewidth=2, label="Velocidade Darcy (Analítica)")

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$u$ [m/s]")
plt.xlim(x_cells.min(), x_cells.max())

# força notação científica elegante
ax = plt.gca()
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax.yaxis.get_offset_text().set_fontsize(10)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/DN/k=1e-16/estacionario/velocity-comparison.png")




# =========================
# Plot pressão
# =========================
# """
plt.figure(dpi=300, figsize=(8, 6))

plt.plot(x_values, p_values, "o", markersize=3, label="FEM (CG1)")
plt.plot(x_values, p_analytical, "-", linewidth=2, label="Solução analítica")

plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [MPa]")
plt.xlim(x_values.min(), x_values.max())

# ESCALA BOA
plt.ylim(19.9, 20.1)  # ajusta para perto de 20 MPa

# REMOVE offset tipo +2e1
ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/DN/k=1e-16/estacionario/pressure-comparison.png")
