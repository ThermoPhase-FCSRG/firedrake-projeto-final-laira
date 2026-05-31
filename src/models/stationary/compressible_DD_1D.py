"""
Descrição do problema: (estacionário)
----------------------

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é compressivel,
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada é:

    (k/μ) ∂/∂x ( p ∂p/∂x ) = 0

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)

Nesta versão:  
- mu depende de p e T (com T constante)
- Z depende de p e T (com T constante) 

rodar com: python -m src.models.stationary.compressible_DD_1D
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import FIGURES_SIM
from src.interpolation.interpolators import load_interpolators 

# =========================
# Mesh definition
numel = 100 # mudei de 200 para 100 (!!!)
L = 200.0   # alterado de 50 para 200m para ver melhor a evolução da pressão
mesh = IntervalMesh(numel, 0.0, L)

# =========================
# Function space
degree = 1
V = FunctionSpace(mesh, "CG", degree)

# =========================
# Boundary conditions
p_left = Constant(2.0e7)   # 200 bar
p_right = Constant(1.0e7)  # 100 bar

bc_left = DirichletBC(V, p_left, 1)
bc_right = DirichletBC(V, p_right, 2)

bcs = [bc_left, bc_right]

# =========================
# Unknown and test function
p = Function(V, name="Pressure")
v = TestFunction(V)

# =========================
# Initial guess
p.assign(1.5e7)  # Define um valor inicial para o método de Newton começar a iteração.

# =========================
# Physical parameters
# kappa = Constant(1.0e-18)
kappa = Constant(1.0e-16)   # =0.0101325 mD TESTE (!!!)
# mu = Constant(0.94e-5)
mu_field = Function(V, name="Viscosity")  
T_const = 300.0  # temperatura fixa (por enquanto!!!)

z_field = Function(V, name="Compressibility")   # Compressibility 

def fp(p): 
    return p / z_field

# =========================
# Interpoladores
Z_interp, rho_interp, mu_interp = load_interpolators()

# =========================
# Variational formulation (STEADY STATE)
F = (kappa / mu_field) * inner(fp(p) * grad(p), grad(v)) * dx

# =========================
# Solver parameters
solver_parameters = {
    "mat_type": "aij",  # matriz esparsa padrão
    "snes_type": "newtonls",   # Método de Newton com linha de busca
    "pc_type": "lu"  # resolve o sistema linear com fatoração LU
}


# =========================
# PICARD LOOP (ACOPLAMENTO)
# =========================
max_iter = 20
tol = 1e-6

# inicializa com valor constante (chute inicial)
mu_field.assign(0.94e-5)
z_field.assign(1.0) # compressibilidade do gás ideal


for k in range(max_iter):
    print(f"\n--- Iteração de Picard {k+1} ---")

    p_old = p.copy(deepcopy=True)     # guarda solução anterior

    # resolve PDE com mu e z fixos (Newton entra aqui):
    solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)

    # =========================
    # Atualização de μ(P)

    # pega valores nodais da pressão
    p_vals = p.dat.data_ro

    # calcula nova viscosidade via interpolador
    mu_vals = np.array([
        mu_interp([val, T_const])[0] for val in p_vals
    ])

    # =========================
    # Atualização de Z(P)
    z_vals = np.array([
        Z_interp([val, T_const])[0] for val in p_vals
    ])

    # =========================
    # conferindo
    # =========================
    print(f"mu_min = {mu_vals.min():.3e}")
    print(f"mu_max = {mu_vals.max():.3e}")
    print(f"delta mu = {np.linalg.norm(mu_vals - mu_field.dat.data):.3e}")

    print(f"z_min = {z_vals.min():.3e}")
    print(f"z_max = {z_vals.max():.3e}")

    # =========================
    # Atualização dos campos de mu e z para a próxima iteração
    # =========================
    mu_field.dat.data[:] = mu_vals     
    z_field.dat.data[:] = z_vals

    # =========================
    # Critério de convergência
    diff = np.linalg.norm(p.dat.data - p_old.dat.data)

    print(f"Erro = {diff:.3e}")

    if diff < tol:
        print("Convergiu!")
        break

"""
# =========================
# Diagnóstico da pressão
# =========================
p_values_raw = p.dat.data_ro

p_min = p_values_raw.min()
p_max = p_values_raw.max()

print("\n=== Intervalo de pressão no domínio ===")
print(f"p_min = {p_min:.3e} Pa")
print(f"p_max = {p_max:.3e} Pa")
"""

# =========================
# Post-processing 
V_u = FunctionSpace(mesh, "DG", 0) # Function space for velocity
u = Function(V_u, name="Darcy velocity")
u_expr = -(kappa / mu_field) * p.dx(0)
u.project(u_expr)

u_values = u.dat.data_ro.copy() # Pega os valores da velocidade de Darcy
x = SpatialCoordinate(mesh)
x_cell = Function(V_u)
x_cell.project(x[0])

x_cells = x_cell.dat.data_ro.copy()

# =========================
# solution
x_values = mesh.coordinates.dat.data_ro # Pega os valores das coordenadas dos nós da malha.
p_values = p.dat.data_ro / 1e3  # Pega os valores da pressão numérica e converte de Pa para kPa.



# =========================
# Solução analítica (Z = 1)
# =========================

pw = float(p_left)
pr = float(p_right)

# solução analítica em Pa
p_analytical = np.sqrt(
    pw**2 + (pr**2 - pw**2) * x_values / L
)

p_analytical = p_analytical / 1e3 # converter para kPa (igual ao numérico)



# =========================
# Plotting
# =========================

# Plot da pressão numérica
plt.figure(dpi=300, figsize=(8, 6))
plt.plot(x_values, p_values, label="Steady state")
plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [KPa]")
plt.xlim(x_values.min(), x_values.max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-pressure-PICARD.png")
# plt.show()

# =========================
# Plot da velocidade numérica
plt.figure(dpi=300, figsize=(8, 6))
plt.step(x_cells, u_values, where="mid", linewidth=2, label="Velocidade Darcy (FEM)")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$u$ [m/s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "compressible-steady-DD-velocity-PICARD.png")

# =========================
# Plot da comparação entre numérico e analítico
plt.figure(dpi=300, figsize=(8, 6))

# solução numérica (com Z variável)
plt.plot(
    x_values,
    p_values,
    "o",
    markersize=3,
    label="FEM (Z variável)"
)

# solução analítica (Z = 1)
plt.plot(
    x_values,
    p_analytical,
    "-",
    linewidth=2,
    label="Analítica (Z = 1)"
)

plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [kPa]")
plt.xlim(x_values.min(), x_values.max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_SIM / "comparison_Z_vs_ideal.png")