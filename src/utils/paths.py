from pathlib import Path # importa uma biblioteca padrão do Python chamada pathlib

# raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[2]

# pastas principais
FIGURES_DIR = ROOT_DIR / "figures"  # cria um caminho para a pasta "figures" dentro do diretório raiz do projeto
SRC_DIR = ROOT_DIR / "src"  # cria um caminho para a pasta "src
DATA_DIR = ROOT_DIR / "data" # cria um caminho para a pasta "data" dentro do diretório raiz do projeto

# subpastas
# subpastas organizadas
FIGURES_SIM = FIGURES_DIR / "simulations"
FIGURES_PROP = FIGURES_DIR / "properties"
FIGURES_VAL = FIGURES_DIR / "validation" 

# propriedades
FIGURES_PROP_DENSITY = FIGURES_PROP / "density"
FIGURES_PROP_Z = FIGURES_PROP / "z"
FIGURES_PROP_VISCOSITY = FIGURES_PROP / "viscosity"


# garante que existem
FIGURES_SIM.mkdir(parents=True, exist_ok=True)
FIGURES_PROP.mkdir(parents=True, exist_ok=True)
FIGURES_VAL.mkdir(parents=True, exist_ok=True)

FIGURES_PROP_DENSITY.mkdir(parents=True, exist_ok=True)
FIGURES_PROP_Z.mkdir(parents=True, exist_ok=True)
FIGURES_PROP_VISCOSITY.mkdir(parents=True, exist_ok=True)