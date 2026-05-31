# grid.py

import numpy as np

def create_grid(): # criando uma grade de pontos para interpolação
    P_vals = np.linspace(1e7, 2e7, 100) # 100 pontos de pressão entre 10 MPa e 20 MPa
    T_vals = np.linspace(200, 500, 50)  # 50 pontos de temperatura entre 200 K e 500 K
    return P_vals, T_vals