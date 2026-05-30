# src/models/neqsim_model.py

from neqsim.thermo import fluid, TPflash   # TPflash resolve o equilíbrio em T e P. e fluid cria um sistema termodinâmico (mistura + EOS)


def _create_fluid():
    """
    Cria o fluido base (hidrogênio puro)
    """
    fl = fluid("pr")   # Peng-Robinson 
    fl.addComponent("hydrogen", 1.0)  # 1.0 significa 100% de hidrogênio
    fl.setMixingRule("classic")
    fl.init(0)  # inicializa o fluido (calcula propriedades básicas)
    return fl # Retorna o objeto fluido configurado.

def _prepare_fluid(P, T):
    """
    Configura T e P e executa TPflash
    """

    fl = _create_fluid()

    # Pa -> bar
    P_bar = P / 1e5

    fl.setTemperature(T)  # neqsim usa Kelvin
    fl.setPressure(P_bar, "bar")

    TPflash(fl)  # resolve o equilíbrio termodinâmico para as condições dadas, calculando propriedades como densidade, fator Z, viscosidade, etc.

    # inicializa propriedades termodinâmicas mais avançadas
    fl.init(3)

    # inicializa propriedades físicas de transporte (viscosidade, condutividade, difusividade)
    fl.initPhysicalProperties()

    return fl

# ======================================
# DENSIDADE
def calculate_density(P, T):
    fl = _prepare_fluid(P, T)
    rho = fl.getPhase(0).getDensity("kg/m3")  # getPhase(0) retorna a fase gas. (0)é gas; (1) é liquido e (2) é multiplas fases
    return rho


# ======================================
# FATOR Z
def calculate_Z(P, T):

    fl = _prepare_fluid(P, T)

    Z = fl.getPhase(0).getZ()

    return Z


# ======================================
# VISCOSIDADE
def calculate_viscosity(P, T):

    fl = _prepare_fluid(P, T)

    mu = fl.getPhase(0).getPhysicalProperties().getViscosity()

    return mu     # já retorna em Pa.s
