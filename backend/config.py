import os
from pathlib import Path

# Project Roots
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Core Directories
MODELS_DIR = PROJECT_ROOT / "models"
DATABASES_DIR = PROJECT_ROOT / "databases"
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Database Files
PHARMACOPEIA_PATH = DATABASES_DIR / "antibiotics_pharmacopeia.csv"
PATHOGEN_ENVELOPE_PHYSICS_PATH = DATABASES_DIR / "pathogen_envelope_physics.json"
PHARMACOPHORE_CATALOG_PATH = DATABASES_DIR / "pharmacophore_smarts_catalog.json"
BIOMARKER_ONTOLOGY_PATH = DATABASES_DIR / "biomarker_mechanism_ontology.json"
ANTIBIOTIC_SMILES_CHEMISTRY_PATH = DATABASES_DIR / "Antibiotic_SMILES_Chemistry.csv"
INDIAN_CLINICAL_ECONOMICS_PATH = DATABASES_DIR / "Indian_Clinical_Economics_Antibiogram.csv"

# Model Files
ENSEMBLE_WEIGHTS_PATH = MODELS_DIR / "amr_ensemble_weights.pkl"
TABULAR_ENSEMBLE_PATH = MODELS_DIR / "amr_tabular_ensemble.pkl"

# Verify critical paths exist on startup
def verify_paths():
    missing = []
    for path in [
        PHARMACOPEIA_PATH,
        PATHOGEN_ENVELOPE_PHYSICS_PATH,
        PHARMACOPHORE_CATALOG_PATH,
        BIOMARKER_ONTOLOGY_PATH,
        ENSEMBLE_WEIGHTS_PATH,
    ]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        print(f"WARNING: Missing critical files: {missing}")
        return False
    return True

