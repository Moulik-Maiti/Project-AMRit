from config import ANTIBIOTIC_SMILES_CHEMISTRY_PATH, INDIAN_CLINICAL_ECONOMICS_PATH, PATHOGEN_ENVELOPE_PHYSICS_PATH
"""
AMrit Data Ingestion & Preprocessing Pipeline (Layer 1 - Bioinformatic Edition)
==============================================================================
1. Intrinsic Resistome Matrix:
   - Hardcoded EUCAST / CLSI expert intrinsic resistance baseline rules.
   - Prevents AI false-susceptibility errors for intrinsically resistant pathogen-drug pairs.

2. Multi-Gene Variant Aggregator:
   - Evaluates combinations of mutations (e.g. gyrA + parC + blaNDM-1 + ompK36).
   - Computes epistatic synergy rules and Magiorakos et al. MDR / XDR / PDR classifications.

3. FASTQ Minor-Allele Frequency (MAF / VAF %) & Heteroresistance Engine:
   - High-throughput pileup builder over resistance pocket targets.
   - Calculates Depth (DP), Allele counts, and Variant Allele Frequency (VAF %).
   - Flags minority heteroresistant subpopulations (5% - 95% VAF) for ICU warning.

4. 6-Frame Dynamic Motif Pocket Locator (Biopython):
   - Scans full genomes (e.g. 4.68MB) in forward & reverse complement strands.
   - Extracts gyrA QRDR, rpoB RRDR, katG, and parC targets.

5. RDKit Cheminformatics & Pharmacoeconomics:
   - 2048-bit Morgan Fingerprinting, Tanimoto cross-resistance filters, ICMR economics.
"""

import io
import re
import os
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Bioinformatic imports
from Bio import SeqIO
from Bio.Seq import Seq

# Cheminformatics imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs


# =====================================================================
# 1. INTRINSIC RESISTOME & ENVELOPE BIOPHYSICS ENGINE
# =====================================================================

class IntrinsicResistomeEngine:
    """Enforces biological intrinsic resistance rules from curated envelope databases."""

    _cached_envelope_physics = None

    @classmethod
    def _get_envelope_physics(cls) -> Dict[str, Any]:
        if cls._cached_envelope_physics is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = str(PATHOGEN_ENVELOPE_PHYSICS_PATH)
            if os.path.exists(db_path):
                import json
                with open(db_path, "r") as f:
                    cls._cached_envelope_physics = json.load(f)
            else:
                cls._cached_envelope_physics = {}
        return cls._cached_envelope_physics

    @classmethod
    def check_intrinsic_resistance(cls, pathogen: str, antibiotic: str) -> Optional[Dict[str, str]]:
        env_db = cls._get_envelope_physics()
        for norm_pathogen, env_data in env_db.items():
            if norm_pathogen.lower() in pathogen.lower():
                intrinsic_dict = env_data.get("intrinsic_mechanisms", {})
                for norm_drug, rule in intrinsic_dict.items():
                    if norm_drug.lower() in antibiotic.lower():
                        return {
                            "is_intrinsically_resistant": True,
                            "pathogen": norm_pathogen,
                            "antibiotic": norm_drug,
                            "mechanism": rule["mechanism"],
                            "reference": rule.get("reference", "EUCAST / CLSI Expert Rules")
                        }
        return None

    @classmethod
    def apply_intrinsic_filter(cls, pathogen: str, susceptibility_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        filtered = susceptibility_results.copy()
        for drug_name, res in filtered.items():
            rule = cls.check_intrinsic_resistance(pathogen, drug_name)
            if rule:
                res["probability_resistant"] = 1.0
                res["status"] = "Resistant"
                res["is_intrinsic"] = True
                res["intrinsic_mechanism"] = rule["mechanism"]
                res["eucast_reference"] = rule["reference"]
        return filtered


# =====================================================================
# 2. MULTI-GENE VARIANT AGGREGATOR & EPISTASIS ENGINE
# =====================================================================

EPISTATIC_RULES = [
    {
        "name": "High-Level Dual QRDR Disruption",
        "genes_required": {"gyrA", "parC"},
        "description": "Double point mutation in GyrA (Codon 83/87) and ParC (Codon 80) completely abolishes water-metal ion bridge.",
        "affected_drugs": ["Ciprofloxacin", "Levofloxacin", "Nalidixic Acid"],
        "synergy_mic_multiplier": 8.0,
        "clinical_impact": "High-level fluoroquinolone resistance (MIC >= 32 mg/L); completely refractory to standard oral therapy."
    },
    {
        "name": "Pan-Beta-Lactam Hydrolysis + Aztreonam Inactivation",
        "genes_required": {"blaNDM-1", "blaCTX-M-15"},
        "description": "Metallo-beta-lactamase (NDM-1) hydrolyzes carbapenems while ESBL (CTX-M-15) hydrolyzes monobactams/aztreonam.",
        "affected_drugs": ["Meropenem", "Imipenem", "Ceftriaxone", "Cefotaxime", "Ceftazidime", "Piperacillin", "Amoxicillin"],
        "synergy_mic_multiplier": 16.0,
        "clinical_impact": "Pan-beta-lactam failure. Requires combination Aztreonam-Avibactam or Cefiderocol."
    },
    {
        "name": "Carbapenemase + Porin Loss Impermeability Synergy",
        "genes_required": {"blaNDM-1", "ompK36"},
        "description": "Enzymatic carbapenem hydrolysis amplified by outer membrane OmpK36 porin closure.",
        "affected_drugs": ["Meropenem", "Imipenem"],
        "synergy_mic_multiplier": 32.0,
        "clinical_impact": "Extreme Carbapenem Resistance (MIC > 128 mg/L); rendering carbapenem-based salvage therapy impossible."
    },
    {
        "name": "Colistin Membrane Modification via mcr-1 Plasmid",
        "genes_required": {"mcr-1"},
        "description": "Phosphoethanolamine transferase modifications of lipid A reduce anionic charge.",
        "affected_drugs": ["Colistin"],
        "synergy_mic_multiplier": 10.0,
        "clinical_impact": "Loss of last-resort polymyxin defense. Strict infection control quarantine required."
    }
]

class MultiGeneVariantAggregator:
    """Aggregates multiple genomic markers, calculates epistasis, and classifies MDR/XDR/PDR."""

    ANTIMICROBIAL_CATEGORIES = {
        "Fluoroquinolones": ["Ciprofloxacin", "Levofloxacin", "Nalidixic Acid"],
        "Carbapenems": ["Meropenem", "Imipenem"],
        "Cephalosporins": ["Ceftriaxone", "Cefotaxime", "Ceftazidime"],
        "Penicillins": ["Amoxicillin", "Piperacillin"],
        "Aminoglycosides": ["Amikacin", "Gentamicin"],
        "Polymyxins": ["Colistin"],
        "Glycopeptides": ["Vancomycin", "Teicoplanin"],
        "Tetracyclines": ["Doxycycline", "Minocycline"],
        "Oxazolidinones": ["Linezolid"],
        "Macrolides": ["Azithromycin"],
        "Anti-tubercular": ["Isoniazid", "Rifampicin", "Ethambutol"]
    }

    @classmethod
    def analyze_isolate_variants(cls, detected_variants: List[str], pathogen: str) -> Dict[str, Any]:
        variant_set = set(detected_variants)
        triggered_epistasis = []
        
        # Check epistatic interactions
        for rule in EPISTATIC_RULES:
            reqs = rule["genes_required"]
            # Match gene names within detected variant tokens
            matched_count = sum(1 for req in reqs if any(req.lower() in v.lower() for v in variant_set))
            if matched_count == len(reqs):
                triggered_epistasis.append({
                    "rule_name": rule["name"],
                    "description": rule["description"],
                    "affected_drugs": rule["affected_drugs"],
                    "synergy_mic_multiplier": rule["synergy_mic_multiplier"],
                    "clinical_impact": rule["clinical_impact"]
                })

        # Calculate affected drug profile
        resistant_drugs = set()
        for ep in triggered_epistasis:
            for drug in ep["affected_drugs"]:
                resistant_drugs.add(drug)
        
        # Add single variant contributions
        for v in variant_set:
            if "gyrA" in v or "parC" in v:
                resistant_drugs.update(["Ciprofloxacin", "Levofloxacin", "Nalidixic Acid"])
            if "blaNDM" in v or "blaOXA" in v or "blaKPC" in v:
                resistant_drugs.update(["Meropenem", "Imipenem", "Ceftriaxone", "Cefotaxime", "Piperacillin"])
            if "rpoB" in v:
                resistant_drugs.add("Rifampicin")
            if "katG" in v:
                resistant_drugs.add("Isoniazid")
            if "mcr" in v:
                resistant_drugs.add("Colistin")
            if "mecA" in v:
                resistant_drugs.update(["Amoxicillin", "Cefotaxime", "Piperacillin"])
            if "vanA" in v or "vanB" in v:
                resistant_drugs.update(["Vancomycin", "Teicoplanin"])

        # Add intrinsic resistance
        for drug_cat, drug_list in cls.ANTIMICROBIAL_CATEGORIES.items():
            for drug in drug_list:
                if IntrinsicResistomeEngine.check_intrinsic_resistance(pathogen, drug):
                    resistant_drugs.add(drug)

        # Magiorakos Consensus AMR Classification
        resistant_categories = set()
        total_categories = len(cls.ANTIMICROBIAL_CATEGORIES)
        
        for cat, drugs in cls.ANTIMICROBIAL_CATEGORIES.items():
            if any(d in resistant_drugs for d in drugs):
                resistant_categories.add(cat)

        num_res_cats = len(resistant_categories)
        if num_res_cats >= total_categories:
            classification = "PDR (Pan-Drug Resistant)"
        elif num_res_cats >= (total_categories - 2):
            classification = "XDR (Extensively Drug Resistant)"
        elif num_res_cats >= 3:
            classification = "MDR (Multi-Drug Resistant)"
        else:
            classification = "Non-MDR / Susceptible Profile"

        return {
            "pathogen": pathogen,
            "detected_variants": detected_variants,
            "epistatic_synergies": triggered_epistasis,
            "predicted_resistant_drugs": sorted(list(resistant_drugs)),
            "resistant_drug_categories": sorted(list(resistant_categories)),
            "amr_epidemiological_classification": classification,
            "magiorakos_criteria_score": f"{num_res_cats}/{total_categories} antimicrobial classes non-susceptible"
        }


# =====================================================================
# 3. FASTQ PILEUP & HETERORESISTANCE (VAF %) ENGINE
# =====================================================================

class FASTQHeteroresistanceEngine:
    """Builds base pileup across FASTQ reads and calculates Variant Allele Frequency (VAF %)."""

    @classmethod
    def compute_codon_vaf(cls, fastq_reads: List[str], target_pocket: str = "gyrA_QRDR") -> Dict[str, Any]:
        if not fastq_reads:
            return {"error": "No FASTQ reads provided"}
        
        # GyrA QRDR codon 83 anchor motif: CATGGTGACT[CG/TG]GCGGTCTAT
        # Codon 83 is preceded by CATGGTGACT (10bp)
        motif_prefix = "CATGGTGACT"
        codon_counts = {}
        total_depth = 0

        for read in fastq_reads:
            read_upper = read.upper()
            pos = read_upper.find(motif_prefix)
            if pos != -1 and len(read_upper) >= pos + len(motif_prefix) + 3:
                codon_dna = read_upper[pos + len(motif_prefix) : pos + len(motif_prefix) + 3]
                if set(codon_dna).issubset(set("ACGT")):
                    codon_counts[codon_dna] = codon_counts.get(codon_dna, 0) + 1
                    total_depth += 1

        if total_depth == 0:
            return {
                "target": target_pocket,
                "depth_coverage": 0,
                "message": "Insufficient coverage over target resistance locus"
            }

        alleles = []
        for codon, count in sorted(codon_counts.items(), key=lambda x: -x[1]):
            vaf_pct = (count / total_depth) * 100.0
            try:
                aa = str(Seq(codon).translate())
            except Exception:
                aa = "?"
            
            # Clinical interpretation
            if vaf_pct >= 95.0:
                interp = "Homogeneous Dominant Allele"
            elif vaf_pct >= 5.0:
                interp = "⚠️ Heteroresistant Minority Subpopulation (High ICU relapse risk)"
            else:
                interp = "Sequencing Noise / Ultra-low frequency"

            alleles.append({
                "codon_dna": codon,
                "translated_aa": aa,
                "read_count": count,
                "variant_allele_frequency_pct": round(vaf_pct, 2),
                "clinical_heteroresistance_status": interp
            })

        has_heteroresistance = any(5.0 <= a["variant_allele_frequency_pct"] < 95.0 for a in alleles)

        return {
            "target": target_pocket,
            "total_depth_coverage_dp": total_depth,
            "heteroresistance_detected": has_heteroresistance,
            "alleles": alleles,
            "clinical_guidance": (
                "CRITICAL ALERT: Minority resistant subpopulation detected. Standard AST may yield false susceptible result; rapid selective expansion under monotherapy anticipated."
                if has_heteroresistance else "Homogeneous isolate population."
            )
        }


# =====================================================================
# 4. 6-FRAME DYNAMIC MOTIF POCKET LOCATOR
# =====================================================================

POCKET_SIGNATURES = {
    "gyrA_QRDR": {
        "gene": "gyrA",
        "description": "DNA Gyrase QRDR (Fluoroquinolone target, Codon 83/87)",
        "aa_pattern": r"HGD[A-Z]{2,4}Y[A-Z]{2}IVR",
        "wt_codon_offset": 3,
        "flank_upstream_aa": 15,
        "flank_downstream_aa": 35
    },
    "rpoB_RRDR": {
        "gene": "rpoB",
        "description": "RNA Polymerase Beta-Subunit RRDR (Rifampicin target, Codon 450)",
        "aa_pattern": r"[LIVMF]FD[A-Z]{2}R[A-Z]{2}G[A-Z]{2}K[LIVMF]R|QTLINIRPVVAAIK",
        "wt_codon_offset": 6,
        "flank_upstream_aa": 15,
        "flank_downstream_aa": 35
    },
    "katG_315": {
        "gene": "katG",
        "description": "Catalase-Peroxidase Activation Pocket (Isoniazid target, Codon 315)",
        "aa_pattern": r"TSGIE[A-Z]{2}WTSTPTKWD|W[A-Z]{2}T[A-Z]P[A-Z]KWD",
        "wt_codon_offset": 1,
        "flank_upstream_aa": 15,
        "flank_downstream_aa": 35
    },
    "parC_QRDR": {
        "gene": "parC",
        "description": "Topoisomerase IV QRDR (Quinolone secondary target, Codon 80)",
        "aa_pattern": r"HGD[A-Z]{2}A[A-Z]{2}VR|MSDMAER[A-Z]{4}ALR",
        "wt_codon_offset": 3,
        "flank_upstream_aa": 15,
        "flank_downstream_aa": 35
    }
}

class GenomicSequenceIngestor:
    """Dynamic Multi-FASTA & FASTQ QC and Resistance Pocket Scanner."""

    IUPAC_DNA_CHARS = set("ACGTNacgtn")

    @classmethod
    def validate_dna(cls, seq_str: str) -> bool:
        clean = "".join(seq_str.split())
        return len(clean) > 0 and set(clean).issubset(cls.IUPAC_DNA_CHARS)

    @classmethod
    def calculate_gc(cls, seq_str: str) -> float:
        seq_upper = seq_str.upper()
        g_count = seq_upper.count("G")
        c_count = seq_upper.count("C")
        total = len(seq_upper)
        return round((g_count + c_count) / total * 100.0, 2) if total > 0 else 0.0

    @classmethod
    def parse_fasta(cls, fasta_content_or_path: Union[str, bytes]) -> List[Dict[str, Any]]:
        if isinstance(fasta_content_or_path, str) and os.path.exists(fasta_content_or_path):
            handle = open(fasta_content_or_path, "r")
        elif isinstance(fasta_content_or_path, str):
            handle = io.StringIO(fasta_content_or_path)
        else:
            handle = io.StringIO(fasta_content_or_path.decode("utf-8", errors="ignore"))

        results = []
        try:
            for record in SeqIO.parse(handle, "fasta"):
                seq_str = str(record.seq).upper()
                is_valid = cls.validate_dna(seq_str)
                gc = cls.calculate_gc(seq_str)
                pockets = cls.extract_resistance_pockets(seq_str)
                
                results.append({
                    "header": record.id or record.description,
                    "length_bp": len(seq_str),
                    "gc_percent": gc,
                    "is_valid_dna": is_valid,
                    "extracted_pockets": pockets
                })
        finally:
            handle.close()

        return results

    @classmethod
    def parse_fastq(cls, fastq_content_or_path: Union[str, bytes], min_qscore: float = 20.0) -> Dict[str, Any]:
        if isinstance(fastq_content_or_path, str) and os.path.exists(fastq_content_or_path):
            handle = open(fastq_content_or_path, "r")
        elif isinstance(fastq_content_or_path, str):
            handle = io.StringIO(fastq_content_or_path)
        else:
            handle = io.StringIO(fastq_content_or_path.decode("utf-8", errors="ignore"))

        records_qc = []
        raw_reads = []
        try:
            for record in SeqIO.parse(handle, "fastq"):
                phred_scores = record.letter_annotations.get("phred_quality", [])
                mean_q = float(np.mean(phred_scores)) if phred_scores else 0.0
                
                if mean_q < min_qscore:
                    continue
                
                seq_str = str(record.seq).upper()
                raw_reads.append(seq_str)
                records_qc.append({
                    "header": record.id,
                    "length_bp": len(seq_str),
                    "mean_phred_quality": round(mean_q, 2)
                })
        finally:
            handle.close()

        # Compute heteroresistance pileup
        heteroresistance_info = FASTQHeteroresistanceEngine.compute_codon_vaf(raw_reads, "gyrA_QRDR")

        return {
            "format": "FASTQ",
            "passed_qc_reads": len(records_qc),
            "mean_coverage_depth": len(raw_reads),
            "heteroresistance_analysis": heteroresistance_info,
            "read_summaries": records_qc[:10]
        }

    @classmethod
    def extract_resistance_pockets(cls, target_sequence: str) -> Dict[str, Dict[str, Any]]:
        found_pockets = {}
        seq_len = len(target_sequence)
        
        if seq_len < 30:
            return found_pockets

        strands = [("+", target_sequence)]
        if seq_len >= 100:
            strands.append(("-", str(Seq(target_sequence).reverse_complement())))

        for pocket_name, sig in POCKET_SIGNATURES.items():
            pattern = sig["aa_pattern"]
            
            for strand, s in strands:
                if pocket_name in found_pockets:
                    break
                
                for frame in range(3):
                    subseq = s[frame:]
                    subseq = subseq[:len(subseq) - (len(subseq) % 3)]
                    if len(subseq) < 30:
                        continue
                    
                    try:
                        trans = str(Seq(subseq).translate())
                    except Exception:
                        continue
                    
                    m = re.search(pattern, trans)
                    if m:
                        aa_pos = m.start()
                        dna_start = max(0, frame + (aa_pos - sig["flank_upstream_aa"]) * 3)
                        dna_end = min(len(s), frame + (aa_pos + sig["flank_downstream_aa"]) * 3)
                        pocket_dna = s[dna_start:dna_end]
                        
                        target_codon_dna = s[frame + (aa_pos + sig["wt_codon_offset"]) * 3 : frame + (aa_pos + sig["wt_codon_offset"] + 1) * 3]
                        target_aa = str(Seq(target_codon_dna).translate()) if len(target_codon_dna) == 3 else "Unknown"

                        found_pockets[pocket_name] = {
                            "gene": sig["gene"],
                            "description": sig["description"],
                            "strand": strand,
                            "frame": frame,
                            "dna_start": dna_start,
                            "dna_end": dna_end,
                            "pocket_len_bp": len(pocket_dna),
                            "target_codon_dna": target_codon_dna,
                            "target_codon_aa": target_aa,
                            "extracted_pocket_seq": pocket_dna
                        }
                        break

        return found_pockets


# =====================================================================
# 5. CHEMINFORMATICS & RDKit FINGERPRINTING
# =====================================================================

class CheminformaticsProcessor:
    """Standardizes SMILES and computes 2048-bit Morgan Fingerprints & descriptors."""

    def __init__(self, chemistry_db_path: Optional[str] = None):
        self.db = {}
        if chemistry_db_path and os.path.exists(chemistry_db_path):
            df = pd.read_csv(chemistry_db_path)
            self.db = df.set_index("Antibiotic_Name").to_dict(orient="index")

    def get_mol(self, smiles_or_drug_name: str) -> Optional[Chem.Mol]:
        if smiles_or_drug_name in self.db:
            smiles = self.db[smiles_or_drug_name]["Canonical_SMILES"]
        else:
            smiles = smiles_or_drug_name
        
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    def get_morgan_fingerprint(self, smiles_or_drug_name: str, n_bits: int = 2048) -> np.ndarray:
        mol = self.get_mol(smiles_or_drug_name)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def get_physicochemical_descriptors(self, smiles_or_drug_name: str) -> Dict[str, float]:
        mol = self.get_mol(smiles_or_drug_name)
        if mol is None:
            return {
                "Molecular_Weight": 0.0,
                "LogP": 0.0,
                "TPSA": 0.0,
                "Num_H_Donors": 0,
                "Num_H_Acceptors": 0
            }
        return {
            "Molecular_Weight": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "Num_H_Donors": int(Descriptors.NumHDonors(mol)),
            "Num_H_Acceptors": int(Descriptors.NumHAcceptors(mol))
        }

    def compute_tanimoto_similarity(self, drug1: str, drug2: str) -> float:
        mol1 = self.get_mol(drug1)
        mol2 = self.get_mol(drug2)
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))


# =====================================================================
# 6. UNIVERSAL INGESTION AUTO-ROUTER
# =====================================================================

VALID_PATHOGENS = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",
    "Mycobacterium tuberculosis",
    "Neisseria gonorrhoeae"
]

class SingleIsolateInput(BaseModel):
    sample_id: Optional[str] = "ISO_PATIENT"
    raw_sequence: Optional[str] = None
    pathogen_species: str = Field(..., description="Target bacterial pathogen")
    detected_variants: Optional[List[str]] = Field(default_factory=list, description="List of detected mutations/genes (e.g. gyrA_S83L, blaNDM-1)")
    tested_antibiotic: Optional[str] = "Ciprofloxacin"
    mic_value_mg_l: Optional[float] = None
    isolate_source: Optional[str] = "clinical"
    geographic_region: Optional[str] = "National Average"
    collection_year: Optional[int] = 2024

    @field_validator("pathogen_species")
    def validate_pathogen(cls, v):
        for p in VALID_PATHOGENS:
            if p.lower() in v.lower():
                return p
        return v


class UniversalIngestRouter:
    """Auto-detects format (FASTA, FASTQ, CSV Antibiogram, JSON) and routes to pipeline."""

    def __init__(self, base_data_dir: str = ''):
        self.base_dir = base_data_dir
        self.chem_path = str(ANTIBIOTIC_SMILES_CHEMISTRY_PATH)
        self.econ_path = str(INDIAN_CLINICAL_ECONOMICS_PATH)
        self.chem_processor = CheminformaticsProcessor(self.chem_path)
        
        self.econ_db = {}
        if os.path.exists(self.econ_path):
            df_econ = pd.read_csv(self.econ_path)
            self.econ_db = df_econ.set_index("Antibiotic_Name").to_dict(orient="index")

    def ingest_payload(self, raw_input: Union[str, bytes, dict, pd.DataFrame]) -> Dict[str, Any]:
        if isinstance(raw_input, dict):
            return self._handle_structured_dict(raw_input)
        
        if isinstance(raw_input, pd.DataFrame):
            return self._handle_dataframe(raw_input)
        
        text_content = raw_input if isinstance(raw_input, str) else raw_input.decode("utf-8", errors="ignore")
        text_stripped = text_content.strip()

        if text_stripped.startswith(">"):
            qc_results = GenomicSequenceIngestor.parse_fasta(text_content)
            return {
                "format": "FASTA",
                "status": "success",
                "record_count": len(qc_results),
                "sequences": qc_results
            }

        if text_stripped.startswith("@"):
            return GenomicSequenceIngestor.parse_fastq(text_content)

        if "," in text_stripped and "\n" in repr(text_stripped):
            try:
                df = pd.read_csv(io.StringIO(text_stripped))
                return self._handle_dataframe(df)
            except Exception:
                pass

        if GenomicSequenceIngestor.validate_dna(text_stripped):
            pockets = GenomicSequenceIngestor.extract_resistance_pockets(text_stripped.upper())
            return {
                "format": "RAW_DNA_SEQUENCE",
                "status": "success",
                "length_bp": len(text_stripped),
                "gc_percent": GenomicSequenceIngestor.calculate_gc(text_stripped),
                "extracted_pockets": pockets
            }

        return {
            "format": "UNKNOWN",
            "status": "error",
            "message": "Unrecognized input format. Supported: FASTA, FASTQ, CSV Antibiogram, or JSON Isolate payload."
        }

    def _handle_structured_dict(self, data: dict) -> Dict[str, Any]:
        validated = SingleIsolateInput(**data)
        pathogen = validated.pathogen_species
        drug = validated.tested_antibiotic
        variants = validated.detected_variants
        
        # 1. Multi-Gene Epistasis Analysis
        multi_gene_analysis = MultiGeneVariantAggregator.analyze_isolate_variants(variants, pathogen)
        
        # 2. Intrinsic Resistance Check
        intrinsic_rule = IntrinsicResistomeEngine.check_intrinsic_resistance(pathogen, drug)
        
        # 3. Chemistry Descriptors
        chem_desc = self.chem_processor.get_physicochemical_descriptors(drug)
        morgan_fp = self.chem_processor.get_morgan_fingerprint(drug, n_bits=2048)
        
        # 4. Economics
        econ_info = self.econ_db.get(drug, {
            "Cost_Per_Dose_INR": 100,
            "Bioavailability_Percent": 50.0,
            "WHO_AWaRe_Category": "Watch",
            "ICMR_Estimated_Resistance_Rate_India": "50.0%"
        })

        return {
            "format": "SINGLE_ISOLATE",
            "status": "success",
            "isolate": validated.model_dump(),
            "intrinsic_resistance_check": intrinsic_rule,
            "multi_gene_epistasis_profile": multi_gene_analysis,
            "chemistry": {
                **chem_desc,
                "morgan_fingerprint_bits": int(np.sum(morgan_fp))
            },
            "pharmacoeconomics": econ_info
        }

    def _handle_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_rows = len(df)
        cols = df.columns.tolist()
        return {
            "format": "BATCH_CSV",
            "status": "success",
            "total_isolates": total_rows,
            "columns": cols,
            "sample_data": df.head(5).to_dict(orient="records")
        }
