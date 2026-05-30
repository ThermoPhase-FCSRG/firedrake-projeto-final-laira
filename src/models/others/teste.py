import numpy as np
from src.interpolation.interpolators import load_interpolators

Z_interp, rho_interp, mu_interp = load_interpolators()

P_test1 = 1e7

print("Z(1e7, 300K) =", Z_interp([P_test1, 300]))
print("Z(1e7, 400K) =", Z_interp([P_test1, 400]))

P = np.load("data/tables/P.npy")
T = np.load("data/tables/T.npy")
Z = np.load("data/tables/Z.npy")

# pega um ponto do grid
i = 10
j = 5

P_test = P[i]
T_test = T[j]

Z_table = Z[i, j]
Z_interp_val = Z_interp([P_test, T_test])[0]

print("Tabela:", Z_table)
print("Interpolado:", Z_interp_val)
print("Diferença:", abs(Z_table - Z_interp_val))