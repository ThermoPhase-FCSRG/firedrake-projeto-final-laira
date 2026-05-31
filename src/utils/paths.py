from pathlib import Path # importa uma biblioteca padrão do Python chamada pathlib

# =====================================
# Diretórios principais
# =====================================

ROOT_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"

# =====================================
# Figuras
# =====================================

FIGURES_PROP = FIGURES_DIR / "properties"
FIGURES_VAL = FIGURES_DIR / "validation"
FIGURES_SIM = FIGURES_DIR / "simulations"

# =====================================
# Propriedades
# =====================================

FIGURES_PROP_DENSITY = FIGURES_PROP / "density"
FIGURES_PROP_Z = FIGURES_PROP / "z"
FIGURES_PROP_VISCOSITY = FIGURES_PROP / "viscosity"

# =====================================
# Simulações IDEAIS
# =====================================

FIGURES_SIM_IDEAL = FIGURES_SIM / "ideal"

FIGURES_SIM_IDEAL_STEADY_DD = FIGURES_SIM_IDEAL / "steady_DD"
FIGURES_SIM_IDEAL_STEADY_DN = FIGURES_SIM_IDEAL / "steady_DN"

FIGURES_SIM_IDEAL_TRANSIENT_DD = FIGURES_SIM_IDEAL / "transient_DD"
FIGURES_SIM_IDEAL_TRANSIENT_DN = FIGURES_SIM_IDEAL / "transient_DN"

# =====================================
# Simulações COMPRESSÍVEIS
# =====================================

FIGURES_SIM_COMPRESSIBLE = FIGURES_SIM / "compressible"

FIGURES_SIM_COMPRESSIBLE_STEADY_DD = (
    FIGURES_SIM_COMPRESSIBLE / "steady_DD"
)

FIGURES_SIM_COMPRESSIBLE_STEADY_DN = (
    FIGURES_SIM_COMPRESSIBLE / "steady_DN"
)

FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DD = (
    FIGURES_SIM_COMPRESSIBLE / "transient_DD"
)

FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DN = (
    FIGURES_SIM_COMPRESSIBLE / "transient_DN"
)

ALL_DIRS = [
    FIGURES_PROP,
    FIGURES_VAL,
    FIGURES_SIM,

    FIGURES_PROP_DENSITY,
    FIGURES_PROP_Z,
    FIGURES_PROP_VISCOSITY,

    FIGURES_SIM_IDEAL_STEADY_DD,
    FIGURES_SIM_IDEAL_STEADY_DN,
    FIGURES_SIM_IDEAL_TRANSIENT_DD,
    FIGURES_SIM_IDEAL_TRANSIENT_DN,

    FIGURES_SIM_COMPRESSIBLE_STEADY_DD,
    FIGURES_SIM_COMPRESSIBLE_STEADY_DN,
    FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DD,
    FIGURES_SIM_COMPRESSIBLE_TRANSIENT_DN,
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)