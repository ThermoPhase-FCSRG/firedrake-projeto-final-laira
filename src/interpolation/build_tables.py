# build_tables.py

# Script para construir as tabelas de propriedades do hidrogênio usando CoolProp
# As tabelas serão salvas em formato .npy para uso posterior na interpolação
# transforma um domínio termodinâmico contínuo em uma tabela discreta

import numpy as np
# from src.properties.coolprop_model import calculate_Z, calculate_density, calculate_viscosity 
from src.properties.neqsim_model import calculate_Z, calculate_density, calculate_viscosity 
# from src.properties.thermo_model import calculate_Z, calculate_density, calculate_viscosity 
# from src.properties.thermo_pr_model import calculate_Z, calculate_density, calculate_viscosity 
# from src.properties.thermopack_model import calculate_Z, calculate_density, calculate_viscosity 


from src.interpolation.grid import create_grid
import os


def build():
    print("Criando grid...")
    P_vals, T_vals = create_grid()

    print("Inicializando matrizes...")   
    Z_grid = np.zeros((len(P_vals), len(T_vals)))
    rho_grid = np.zeros((len(P_vals), len(T_vals)))
    mu_grid = np.zeros((len(P_vals), len(T_vals)))

    print("Calculando propriedades...")
    for i, P in enumerate(P_vals):
        for j, T in enumerate(T_vals):
            Z_grid[i, j] = calculate_Z(P, T)
            rho_grid[i, j] = calculate_density(P, T)
            mu_grid[i, j] = calculate_viscosity(P, T)

    print("Criando pasta data/tables se não existir...")
    os.makedirs("data/tables", exist_ok=True)

    print("Salvando tabelas...")
    np.save("data/tables/Z.npy", Z_grid)
    np.save("data/tables/rho.npy", rho_grid)
    np.save("data/tables/mu.npy", mu_grid)
    np.save("data/tables/P.npy", P_vals)
    np.save("data/tables/T.npy", T_vals)

if __name__ == "__main__":  # Só execute build() se esse arquivo for o programa principal
        build()