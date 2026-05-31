# interpolators.py

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def load_interpolators():
    P = np.load("data/tables/P.npy")
    T = np.load("data/tables/T.npy")

    Z = np.load("data/tables/Z.npy")
    rho = np.load("data/tables/rho.npy")
    mu = np.load("data/tables/mu.npy")

    Z_interp = RegularGridInterpolator((P, T), Z)  # Cria um interpolador para Z
    rho_interp = RegularGridInterpolator((P, T), rho)
    mu_interp = RegularGridInterpolator((P, T), mu)

    return Z_interp, rho_interp, mu_interp