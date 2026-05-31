# coolprop.py

from CoolProp.CoolProp import PropsSI

# Constante do gás para hidrogênio (J/kg.K)
R = 4124  # J/kg.K


def calculate_Z(P, T):
    """
    Calcula o fator de compressibilidade Z.
    [Z] = adimensional
    """
    rho = calculate_density(P, T)
    Z = P / (rho * R * T)
    return Z

def calculate_density(P, T):
    """  
    Calcula a densidade do hidrogênio
    [rho] = kg/m³  
    P: pressão (Pa)
    T: temperatura (K)
    """
    rho = PropsSI("D", "P", P, "T", T, "Hydrogen") 
    return rho

def calculate_viscosity(P, T):
    """
    Calcula a viscosidade do hidrogênio
    [mu] = Pa·s
    """

    mu = PropsSI("V", "P", P, "T", T, "Hydrogen")   # "V" viscosidade dinâmica (Pa·s)
    return mu