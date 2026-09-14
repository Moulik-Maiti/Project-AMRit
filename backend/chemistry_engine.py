from config import PHARMACOPEIA_PATH, PATHOGEN_ENVELOPE_PHYSICS_PATH, PHARMACOPHORE_CATALOG_PATH, BIOMARKER_ONTOLOGY_PATH
"""
AMrit Chemistry & Molecular Mechanics Engine (Layer 1 & 2)
==========================================================
Cheminformatics-driven target recognition, envelope biophysics,
pharmacophore SMARTS matching, and resistance gene inactivation.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import DataStructs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATABASES_DIR = os.path.join(PROJECT_DIR, "databases")

SPECIES_BASELINE_MIC = {
    "Escherichia coli": {
        "Ciprofloxacin": 0.015, "Levofloxacin": 0.03, "Nalidixic Acid": 2.0,
        "Meropenem": 0.03, "Imipenem": 0.12, "Ceftriaxone": 0.06, "Cefotaxime": 0.06,
        "Ceftazidime": 0.12, "Piperacillin": 2.0, "Amoxicillin": 2.0, "Amikacin": 1.5,
        "Gentamicin": 0.5, "Colistin": 0.25, "Doxycycline": 0.5, "Minocycline": 0.5,
        "Azithromycin": 2.0, "Linezolid": 64.0, "Vancomycin": 64.0, "Teicoplanin": 64.0,
        "Rifampicin": 8.0, "Isoniazid": 64.0, "Ethambutol": 64.0
    },
    "Klebsiella pneumoniae": {
        "Ciprofloxacin": 0.03, "Levofloxacin": 0.06, "Nalidixic Acid": 2.0,
        "Meropenem": 0.06, "Imipenem": 0.25, "Ceftriaxone": 0.06, "Cefotaxime": 0.06,
        "Ceftazidime": 0.12, "Piperacillin": 4.0, "Amoxicillin": 32.0, "Amikacin": 1.0,
        "Gentamicin": 0.5, "Colistin": 0.25, "Doxycycline": 0.5, "Minocycline": 0.5,
        "Azithromycin": 4.0, "Linezolid": 64.0, "Vancomycin": 64.0, "Teicoplanin": 64.0,
        "Rifampicin": 16.0, "Isoniazid": 64.0, "Ethambutol": 64.0
    },
    "Pseudomonas aeruginosa": {
        "Ciprofloxacin": 0.25, "Levofloxacin": 0.5, "Nalidixic Acid": 64.0,
        "Meropenem": 0.5, "Imipenem": 1.0, "Ceftazidime": 1.5, "Ceftriaxone": 64.0,
        "Cefotaxime": 64.0, "Piperacillin": 4.0, "Amoxicillin": 64.0, "Amikacin": 2.0,
        "Gentamicin": 1.0, "Colistin": 1.0, "Doxycycline": 64.0, "Minocycline": 64.0,
        "Azithromycin": 64.0, "Linezolid": 64.0, "Vancomycin": 64.0, "Teicoplanin": 64.0,
        "Rifampicin": 32.0, "Isoniazid": 64.0, "Ethambutol": 64.0
    },
    "Acinetobacter baumannii": {
        "Ciprofloxacin": 0.5, "Levofloxacin": 0.5, "Nalidixic Acid": 32.0,
        "Meropenem": 0.5, "Imipenem": 0.5, "Ceftazidime": 4.0, "Ceftriaxone": 16.0,
        "Cefotaxime": 16.0, "Piperacillin": 8.0, "Amoxicillin": 64.0, "Amikacin": 2.0,
        "Gentamicin": 1.0, "Colistin": 0.5, "Doxycycline": 1.0, "Minocycline": 0.5,
        "Azithromycin": 64.0, "Linezolid": 64.0, "Vancomycin": 64.0, "Teicoplanin": 64.0,
        "Rifampicin": 2.0, "Isoniazid": 64.0, "Ethambutol": 64.0
    },
    "Staphylococcus aureus": {
        "Vancomycin": 0.75, "Teicoplanin": 0.5, "Linezolid": 1.0, "Doxycycline": 0.25,
        "Minocycline": 0.12, "Ciprofloxacin": 0.25, "Levofloxacin": 0.25, "Gentamicin": 0.25,
        "Amikacin": 2.0, "Azithromycin": 0.5, "Rifampicin": 0.015, "Amoxicillin": 16.0,
        "Piperacillin": 16.0, "Cefotaxime": 2.0, "Ceftriaxone": 2.0, "Ceftazidime": 8.0,
        "Meropenem": 0.25, "Imipenem": 0.06, "Colistin": 64.0, "Nalidixic Acid": 64.0,
        "Isoniazid": 64.0, "Ethambutol": 64.0
    },
    "Mycobacterium tuberculosis": {
        "Isoniazid": 0.05, "Rifampicin": 0.12, "Ethambutol": 1.0, "Levofloxacin": 0.5,
        "Amikacin": 0.5, "Linezolid": 0.5, "Ciprofloxacin": 1.0, "Meropenem": 8.0,
        "Imipenem": 8.0, "Amoxicillin": 64.0, "Cefotaxime": 64.0, "Ceftriaxone": 64.0,
        "Ceftazidime": 64.0, "Piperacillin": 64.0, "Colistin": 64.0, "Gentamicin": 4.0,
        "Doxycycline": 16.0, "Minocycline": 8.0, "Azithromycin": 16.0, "Vancomycin": 64.0,
        "Teicoplanin": 64.0, "Nalidixic Acid": 64.0
    },
    "Neisseria gonorrhoeae": {
        "Ceftriaxone": 0.008, "Azithromycin": 0.25, "Ciprofloxacin": 0.015, "Levofloxacin": 0.03,
        "Doxycycline": 0.25, "Minocycline": 0.25, "Gentamicin": 4.0, "Cefotaxime": 0.015,
        "Amoxicillin": 0.5, "Piperacillin": 0.5, "Meropenem": 0.06, "Imipenem": 0.12,
        "Colistin": 64.0, "Vancomycin": 64.0, "Teicoplanin": 64.0, "Linezolid": 1.0,
        "Rifampicin": 0.25, "Isoniazid": 64.0, "Ethambutol": 64.0, "Amikacin": 16.0,
        "Nalidixic Acid": 4.0, "Ceftazidime": 0.06
    }
}

class CheminformaticsMolecularEngine:
    """Dynamic Cheminformatics, Pharmacophore, and Envelope Permeability Engine."""

    def __init__(self, db_dir: str = DATABASES_DIR):
        self.db_dir = db_dir
        self.pharmacopeia_df = pd.DataFrame()
        self.envelope_physics = {}
        self.pharmacophore_catalog = {}
        self.biomarker_ontology = {}
        self._smarts_mol_cache = {}
        self._drug_mol_cache = {}
        self.load_databases()

    def load_databases(self):
        """Loads all structured reference databases dynamically from JSON/CSV files."""
        pharm_path = str(PHARMACOPEIA_PATH)
        if os.path.exists(pharm_path):
            self.pharmacopeia_df = pd.read_csv(pharm_path)
        
        env_path = str(PATHOGEN_ENVELOPE_PHYSICS_PATH)
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                self.envelope_physics = json.load(f)

        smarts_path = str(PHARMACOPHORE_CATALOG_PATH)
        if os.path.exists(smarts_path):
            with open(smarts_path, "r") as f:
                self.pharmacophore_catalog = json.load(f)
            # Compile SMARTS patterns
            for key, pdata in self.pharmacophore_catalog.items():
                smarts_str = pdata.get("smarts", "")
                if smarts_str and not smarts_str.startswith("MW"):
                    mol_smarts = Chem.MolFromSmarts(smarts_str)
                    if mol_smarts:
                        self._smarts_mol_cache[key] = mol_smarts

        bio_path = str(BIOMARKER_ONTOLOGY_PATH)
        if os.path.exists(bio_path):
            with open(bio_path, "r") as f:
                self.biomarker_ontology = json.load(f)

        # Pre-cache RDKit Mols for pharmacopeia
        for _, row in self.pharmacopeia_df.iterrows():
            drug_name = str(row["Antibiotic_Name"]).strip()
            smi = str(row["Canonical_SMILES"]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol)
                self._drug_mol_cache[drug_name] = mol

    def get_mol(self, drug_name: str) -> Optional[Chem.Mol]:
        if drug_name in self._drug_mol_cache:
            return self._drug_mol_cache[drug_name]
        row = self.pharmacopeia_df[self.pharmacopeia_df["Antibiotic_Name"].str.lower() == drug_name.lower()]
        if not row.empty:
            smi = str(row["Canonical_SMILES"].iloc[0]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol)
                self._drug_mol_cache[drug_name] = mol
                return mol
        return None

    def compute_molecular_descriptors(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Calculates full 2D/3D physicochemical descriptors dynamically."""
        if mol is None:
            return {}
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        return {
            "molecular_weight": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": int(hbd),
            "hba": int(hba),
            "rotatable_bonds": int(rot_bonds),
            "fraction_csp3": round(fraction_csp3, 3),
            "net_formal_charge": int(formal_charge),
            "heavy_atom_count": int(heavy_atoms),
            "aromatic_ring_count": int(aromatic_rings)
        }

    def identify_pharmacophores(self, mol: Chem.Mol) -> List[str]:
        """Dynamically identifies all pharmacophoric substructures using SMARTS matching & molecular rules."""
        if mol is None:
            return []
        
        detected_keys = []
        for key, smarts_mol in self._smarts_mol_cache.items():
            if mol.HasSubstructMatch(smarts_mol):
                detected_keys.append(key)
        
        # Heavy glycopeptide cage rule
        if Descriptors.MolWt(mol) > 1200.0:
            if "glycopeptide_core" not in detected_keys:
                detected_keys.append("glycopeptide_core")
                
        return detected_keys

    def evaluate_envelope_permeability_and_spectrum(
        self, 
        mol: Chem.Mol, 
        drug_name: str, 
        pathogen: str
    ) -> Dict[str, Any]:
        """
        Evaluates biophysical cell envelope permeability, porin size exclusion sieve,
        and intrinsic resistance mechanisms for any pathogen-drug pair.
        """
        descriptors = self.compute_molecular_descriptors(mol)
        mw = descriptors.get("molecular_weight", 400.0)
        pharmacophores = self.identify_pharmacophores(mol)
        
        # Match pathogen in envelope database
        env_record = None
        for p_key, p_val in self.envelope_physics.items():
            if p_key.lower() in pathogen.lower():
                env_record = p_val
                break
        
        if env_record is None:
            env_record = {
                "gram_envelope_type": "Gram-negative",
                "has_outer_membrane": True,
                "porin_size_exclusion_cutoff_da": 600.0,
                "permeability_rules": {"max_mw_da": 650.0, "requires_polyamine_uptake_if_large": True, "mycolic_acid_barrier": False},
                "intrinsic_mechanisms": {}
            }

        # 1. Check Intrinsic Cellular Resistance Mechanisms in database
        intrinsic_mechanisms = env_record.get("intrinsic_mechanisms", {})
        for int_drug, int_info in intrinsic_mechanisms.items():
            if int_drug.lower() in drug_name.lower():
                return {
                    "in_clinical_spectrum": True,
                    "is_permeable": False,
                    "is_intrinsically_resistant": True,
                    "mechanism": int_info["mechanism"],
                    "reference": int_info.get("reference", "EUCAST / CLSI Expert Rules"),
                    "exclusion_reason": f"Intrinsic Resistance: {int_info["mechanism"]}"
                }

        # 2. Biophysical Outer Membrane Porin Size Sieve (Richter-Hergenrother eNTRy Rules)
        has_outer_mem = env_record.get("has_outer_membrane", True)
        porin_cutoff = env_record.get("porin_size_exclusion_cutoff_da", 600.0)
        perm_rules = env_record.get("permeability_rules", {})
        
        if has_outer_mem and mw > porin_cutoff:
            is_polymyxin = "polymyxin_core" in pharmacophores
            if not is_polymyxin:
                return {
                    "in_clinical_spectrum": False,
                    "is_permeable": False,
                    "is_intrinsically_resistant": False,
                    "mechanism": "Steric exclusion at outer-membrane porins",
                    "reference": "Richter & Hergenrother (Nature 2017) Gram-Negative eNTRy Rules",
                    "exclusion_reason": f"Molecular weight ({mw:.1f} Da) exceeds outer-membrane porin cutoff ({porin_cutoff:.0f} Da); excluded by outer envelope sieve."
                }

        # 3. Mycobacterial Lipid Envelope Rules
        if perm_rules.get("mycolic_acid_barrier", False):
            tb_active = any(p in pharmacophores for p in [
                "anti_tb_hydrazide_core", "anti_tb_ethambutol_core", 
                "rifamycin_core", "fluoroquinolone_core", "oxazolidinone_core", "aminoglycoside_core"
            ])
            if not tb_active:
                return {
                    "in_clinical_spectrum": False,
                    "is_permeable": False,
                    "is_intrinsically_resistant": False,
                    "mechanism": "Thick waxy mycolic acid barrier impermeability",
                    "reference": "WHO Consolidated Guidelines on Tuberculosis",
                    "exclusion_reason": "Impermeable to mycolic acid envelope; lacks anti-mycobacterial target binding."
                }

        # 4. Anti-TB Specific Prodrugs on Non-Mycobacterial Species
        if not perm_rules.get("mycolic_acid_barrier", False):
            if "anti_tb_hydrazide_core" in pharmacophores or "anti_tb_ethambutol_core" in pharmacophores:
                return {
                    "in_clinical_spectrum": False,
                    "is_permeable": True,
                    "is_intrinsically_resistant": False,
                    "mechanism": "Narrow-spectrum anti-tubercular agent",
                    "reference": "Clinical Microbiology Spectrum",
                    "exclusion_reason": "Narrow-spectrum anti-tubercular agent (Requires KatG activation / EmbB target absent in non-mycobacteria)."
                }

        # 5. Gram-Positive vs Gram-Negative Specificity
        if env_record.get("gram_envelope_type") == "Gram-positive":
            if "polymyxin_core" in pharmacophores:
                return {
                    "in_clinical_spectrum": False,
                    "is_permeable": False,
                    "is_intrinsically_resistant": True,
                    "mechanism": "Lacks Gram-negative outer-membrane Lipid A target",
                    "reference": "EUCAST Expert Rules v3.3",
                    "exclusion_reason": "Intrinsic inactivity: Gram-positive peptidoglycan lacks outer membrane Lipid A target."
                }
        elif env_record.get("gram_envelope_type") == "Gram-negative":
            if "oxazolidinone_core" in pharmacophores:
                return {
                    "in_clinical_spectrum": False,
                    "is_permeable": False,
                    "is_intrinsically_resistant": False,
                    "mechanism": "Gram-negative intrinsic RND efflux pump extrusion",
                    "reference": "CLSI M100 Ed. 34",
                    "exclusion_reason": "Gram-negative intrinsic multidrug efflux pump extrusion (AcrAB-TolC / MexAB-OprM)."
                }

        return {
            "in_clinical_spectrum": True,
            "is_permeable": True,
            "is_intrinsically_resistant": False,
            "mechanism": None,
            "reference": "Clinically Active Spectrum",
            "exclusion_reason": None
        }

    def parse_variant_token(self, variant_str: str) -> Dict[str, Any]:
        """
        Parses raw genomic variant tokens (e.g. gyrA_S83L, parC_S80I, blaNDM-1, ompK36)
        into exact structured biophysical records and label-encoder compatible fields.
        """
        v = str(variant_str).strip()
        
        # 1. Point Mutation Pattern: e.g. gyrA_S83L, rpoB_S450L, katG_S315T, parC_S80I
        pt_match = re.match(r"^([a-zA-Z0-9_\-]+)[_: ]+([A-Z])?([0-9]+)([A-Z])?", v)
        if pt_match and pt_match.group(3):
            raw_gene = pt_match.group(1)
            wt_aa = pt_match.group(2) or "S"
            codon_num = pt_match.group(3)
            mut_aa = pt_match.group(4) or "L"
            
            norm_gene = raw_gene.lower()
            if "gyra" in norm_gene:
                norm_gene = "gyrA"
                target_reg = f"Codon {codon_num}"
                aro = "ARO:3000211"
                if codon_num == "83":
                    potency = 32.0 if mut_aa == "L" else (48.0 if mut_aa == "W" else 16.0)
                elif codon_num == "87":
                    potency = 16.0 if mut_aa == "N" else (24.0 if mut_aa == "Y" else 14.0)
                else:
                    potency = 12.0
            elif "parc" in norm_gene or "grla" in norm_gene:
                norm_gene = "parC" if "parc" in norm_gene else "grlA"
                target_reg = f"Codon {codon_num}"
                aro = "ARO:3001030"
                potency = 8.0 if mut_aa == "I" else 6.0
            elif "rpob" in norm_gene:
                norm_gene = "rpoB"
                target_reg = f"Codon {codon_num} (RRDR)"
                aro = "ARO:3003740"
                potency = 256.0 if mut_aa == "L" else 128.0
            elif "katg" in norm_gene:
                norm_gene = "katG"
                target_reg = f"Codon {codon_num}"
                aro = "ARO:3003743"
                potency = 128.0 if mut_aa == "T" else 64.0
            elif "embb" in norm_gene:
                norm_gene = "embB"
                target_reg = f"Codon {codon_num}"
                aro = "ARO:3000228"
                potency = 32.0 if mut_aa == "V" else 24.0
            else:
                norm_gene = raw_gene
                target_reg = f"Codon {codon_num}"
                aro = "ARO:3000000"
                potency = 16.0

            return {
                "raw_token": v,
                "gene_marker": norm_gene,
                "mutation_type": "Point Mutation",
                "target_region": target_reg,
                "wt_amino_acid": wt_aa,
                "sample_amino_acid": mut_aa,
                "card_aro_id": aro,
                "potency_multiplier": potency,
                "is_point_mutation": True
            }

        # 2. Porin Mutations / Loss: e.g. ompK35, ompK36, oprD
        v_low = v.lower()
        if "ompk35" in v_low:
            return {
                "raw_token": v, "gene_marker": "ompK35", "mutation_type": "Regulatory/Structural Change",
                "target_region": "Porin", "wt_amino_acid": "WT", "sample_amino_acid": "Lost",
                "card_aro_id": "ARO:3002667", "potency_multiplier": 8.0, "is_point_mutation": False
            }
        if "ompk36" in v_low:
            return {
                "raw_token": v, "gene_marker": "ompK36", "mutation_type": "Regulatory/Structural Change",
                "target_region": "Porin", "wt_amino_acid": "WT", "sample_amino_acid": "Lost",
                "card_aro_id": "ARO:3002668", "potency_multiplier": 12.0, "is_point_mutation": False
            }
        if "oprd" in v_low:
            return {
                "raw_token": v, "gene_marker": "oprD", "mutation_type": "Regulatory/Structural Change",
                "target_region": "Porin", "wt_amino_acid": "WT", "sample_amino_acid": "Lost",
                "card_aro_id": "ARO:3000787", "potency_multiplier": 16.0, "is_point_mutation": False
            }

        # 3. Acquired Beta-Lactamases & Enzymes
        if "ndm" in v_low:
            return {
                "raw_token": v, "gene_marker": "blaNDM-1", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000589", "potency_multiplier": 256.0, "is_point_mutation": False
            }
        if "ctx" in v_low:
            return {
                "raw_token": v, "gene_marker": "blaCTX-M-15", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000557", "potency_multiplier": 128.0, "is_point_mutation": False
            }
        if "oxa-48" in v_low or "oxa48" in v_low:
            return {
                "raw_token": v, "gene_marker": "blaOXA-48", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000078", "potency_multiplier": 64.0, "is_point_mutation": False
            }
        if "kpc" in v_low:
            return {
                "raw_token": v, "gene_marker": "blaKPC-2", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000037", "potency_multiplier": 128.0, "is_point_mutation": False
            }
        if "meca" in v_low:
            return {
                "raw_token": v, "gene_marker": "mecA", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000618", "potency_multiplier": 128.0, "is_point_mutation": False
            }
        if "vana" in v_low:
            return {
                "raw_token": v, "gene_marker": "vanA", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000007", "potency_multiplier": 256.0, "is_point_mutation": False
            }
        if "vanb" in v_low:
            return {
                "raw_token": v, "gene_marker": "vanB", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000008", "potency_multiplier": 128.0, "is_point_mutation": False
            }
        if "mcr" in v_low:
            return {
                "raw_token": v, "gene_marker": "mcr-1", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3003575", "potency_multiplier": 16.0, "is_point_mutation": False
            }
        if "tetm" in v_low:
            return {
                "raw_token": v, "gene_marker": "tetM", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000186", "potency_multiplier": 64.0, "is_point_mutation": False
            }
        if "arma" in v_low or "rmtb" in v_low:
            return {
                "raw_token": v, "gene_marker": "armA", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000596", "potency_multiplier": 256.0, "is_point_mutation": False
            }
        if "ermb" in v_low:
            return {
                "raw_token": v, "gene_marker": "ermB", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000595", "potency_multiplier": 128.0, "is_point_mutation": False
            }
        if "blaz" in v_low:
            return {
                "raw_token": v, "gene_marker": "blaZ", "mutation_type": "Plasmid/Gene Acquisition",
                "target_region": "Full gene", "wt_amino_acid": "Absent", "sample_amino_acid": "Present",
                "card_aro_id": "ARO:3000574", "potency_multiplier": 64.0, "is_point_mutation": False
            }

        return {
            "raw_token": v,
            "gene_marker": v,
            "mutation_type": "Point Mutation",
            "target_region": "Full gene",
            "wt_amino_acid": "WT",
            "sample_amino_acid": "MUT",
            "card_aro_id": "Unknown",
            "potency_multiplier": 8.0,
            "is_point_mutation": False
        }

    def evaluate_all_matching_biomarkers(
        self, 
        mol: Chem.Mol, 
        detected_variants: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all detected genomic variants against the small molecule chemical pharmacophores.
        Returns a list of all matching biophysical mutation records.
        """
        if mol is None or not detected_variants:
            return []

        mol_pharmacophores = set(self.identify_pharmacophores(mol))
        matching_records = []

        for v in detected_variants:
            parsed = self.parse_variant_token(v)
            g_key = parsed["gene_marker"]
            
            ont_rec = None
            for key, rec in self.biomarker_ontology.items():
                if key.lower() in g_key.lower():
                    ont_rec = rec
                    break
            
            if ont_rec:
                targeted_pharm = set(ont_rec.get("targeted_pharmacophores", []))
                if targeted_pharm.intersection(mol_pharmacophores):
                    parsed["ontology_info"] = ont_rec
                    matching_records.append(parsed)

        return matching_records

    def evaluate_biomarker_chemical_inactivation(
        self, 
        mol: Chem.Mol, 
        detected_variants: List[str]
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Evaluates genomic resistance mutations directly against the molecule chemical pharmacophores.
        Returns primary encoded tuple: (Gene_Marker, Mutation_Type, Target_Region, WT_Amino_Acid, Sample_Amino_Acid, CARD_ARO_ID)
        """
        matching = self.evaluate_all_matching_biomarkers(mol, detected_variants)
        if not matching:
            return "WT", "Wild-Type", "Full gene", "WT", "WT", "None"

        primary = max(matching, key=lambda x: x.get("potency_multiplier", 1.0))
        return (
            primary["gene_marker"],
            primary["mutation_type"],
            primary["target_region"],
            primary["wt_amino_acid"],
            primary["sample_amino_acid"],
            primary["card_aro_id"]
        )

    def get_pathogen_baseline_mic(self, pathogen: str, drug_name: str) -> float:
        """Returns the species-specific baseline wild-type MIC (mg/L)."""
        for p_key, drug_dict in SPECIES_BASELINE_MIC.items():
            if p_key.lower() in pathogen.lower():
                return float(drug_dict.get(drug_name, 0.5))
        return 0.5

    def compute_epistatic_permutation_shift(
        self,
        pathogen: str,
        drug_name: str,
        mol: Chem.Mol,
        all_variants: List[str],
        matching_variants: List[Dict[str, Any]],
        ml_predicted_mic: float,
        ml_prob_resistant: float
    ) -> Tuple[float, float, str]:
        """
        Calculates multi-mutation permutation epistasis and continuous biophysical fold shifts.
        Ensures each mutation combination and microbe produces dynamically varying, biologically calibrated MICs.
        """
        base_mic = self.get_pathogen_baseline_mic(pathogen, drug_name)
        
        # 1. If Wild-Type for this drug (0 matching mutations)
        if not matching_variants:
            calibrated_mic = round(float(np.sqrt(ml_predicted_mic * base_mic)), 3)
            calibrated_prob = min(ml_prob_resistant, 0.05)
            return calibrated_mic, calibrated_prob, "Wild-Type Susceptible baseline"

        # 2. Extract matching genes and mutations
        matching_genes = [m["gene_marker"].lower() for m in matching_variants]
        matching_regions = [m["target_region"] for m in matching_variants]
        
        max_single_potency = max(m.get("potency_multiplier", 8.0) for m in matching_variants)
        
        synergy_mult = 1.0
        synergy_desc = []

        # Dual Topoisomerase synergy (gyrA + parC or grlA)
        has_gyra = any("gyra" in g for g in matching_genes)
        has_parc = any("parc" in g or "grla" in g for g in matching_genes)
        if has_gyra and has_parc:
            synergy_mult *= 4.0
            synergy_desc.append("Dual Topoisomerase II/IV Target Destruction")

        # Double GyrA mutation synergy (e.g. S83L + D87N)
        gyra_codons = [m["target_region"] for m in matching_variants if "gyra" in m["gene_marker"].lower()]
        if len(set(gyra_codons)) >= 2:
            synergy_mult *= 3.0
            synergy_desc.append("Double GyrA QRDR Pocket Collapse (Codon 83 + 87)")

        # Metallo-beta-lactamase / Carbapenemase + Porin Loss synergy (blaNDM / blaKPC + ompK36 / oprD)
        has_carbapenemase = any("ndm" in g or "kpc" in g or "oxa-48" in g for g in matching_genes)
        has_porin_loss = any("ompk36" in v.lower() or "oprd" in v.lower() or "ompk35" in v.lower() for v in all_variants)
        if has_carbapenemase and has_porin_loss:
            synergy_mult *= 8.0
            synergy_desc.append("Enzymatic Carbapenemase + Porin Diffusion Barrier Synergy")

        # Dual Beta-Lactamase synergy (blaNDM-1 + blaCTX-M-15)
        has_esbl = any("ctx" in g for g in matching_genes)
        if has_carbapenemase and has_esbl:
            synergy_mult *= 3.0
            synergy_desc.append("Metallo-Beta-Lactamase + ESBL Co-expression")

        # Apply epistatic synergy directly to predicted MIC
        if synergy_mult > 1.0:
            mic_calc = ml_predicted_mic * synergy_mult
        else:
            mic_calc = ml_predicted_mic
        
        final_mic = round(float(np.clip(mic_calc, 0.008, 1024.0)), 3)
        final_prob = float(np.clip(max(ml_prob_resistant, 0.95 + 0.04 * min(1.0, synergy_mult / 8.0)), 0.0, 0.9999))
        
        rationale = " + ".join(synergy_desc) if synergy_desc else f"Target disruption ({matching_variants[0]['gene_marker']} {matching_variants[0]['target_region']})"
        return final_mic, final_prob, rationale
