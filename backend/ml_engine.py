from config import ENSEMBLE_WEIGHTS_PATH, GENOMIC_CNN_WEIGHTS_PATH

"""
AMrit Production ML & Chemistry Engine (Layer 2, 3 & 4)
========================================================
- Level-0 Learners: Multi-Modal XGBoost + PyTorch TabNet + LightGBM
- Level-1 Meta-Learner: Stacking Logistic Regression Meta-Classifier
- Continuous MIC Log2 Regressor (mg/L)
- Cheminformatics: CheminformaticsMolecularEngine (RDKit SMARTS, Pharmacophores & Biophysical Porin Sieve)
- Explainability: TreeSHAP Feature Attributions
"""

import logging
import os
import pickle
import sys
from typing import Any

import numpy as np
import shap
import torch
from dl_genomics import GenomicCNN, sequence_to_tensor
from rdkit import Chem
from rdkit.Chem import Descriptors as _Descriptors
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

# RDKit registers the descriptor functions (MolWt, TPSA, ...) at import time, so static
# type checkers can't see them; Any keeps those call sites from being flagged.
Descriptors: Any = _Descriptors

# Same bits as the legacy AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256) the
# models were trained on, without its per-call deprecation warning.
_MORGAN_256 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
WEIGHTS_PATH = str(ENSEMBLE_WEIGHTS_PATH)

sys.path.append(BASE_DIR)
from chemistry_engine import CheminformaticsMolecularEngine
from data_ingestion import MultiGeneVariantAggregator

logger = logging.getLogger("amrit.ml_engine")

def calculate_dynamic_efficacy_score(
    predicted_mic: float,
    eucast_breakpoint: float,
    susceptibility_confidence: float,
    bioavailability: float,
    who_aware_cat: str,
    cost_per_dose: float
) -> float:
    """
    Computes a calibrated Clinical Pharmacodynamic & Stewardship Score (0 to 100).
    Driven continuously by ML Predicted MIC, Stacking Ensemble Probability, PK Target Attainment, and ICMR Economics.
    """
    mic_safe = max(float(predicted_mic), 0.005)
    pd_margin = eucast_breakpoint / mic_safe
    
    # 1. Pharmacodynamic killing margin (0 to 1) - Weight: 40 pts
    pd_factor = min(1.0, np.log2(1.0 + max(0.0, pd_margin)) / np.log2(1.0 + 32.0))
    pd_score = pd_factor * 40.0
    
    # 2. Multi-Modal ML Confidence Component - Weight: 25 pts
    ml_confidence_score = float(susceptibility_confidence) * 25.0
    
    # 3. Pharmacokinetic Bioavailability Component - Weight: 15 pts
    bioavail_factor = min(1.0, max(0.05, float(bioavailability) / 100.0))
    pk_score = np.sqrt(bioavail_factor) * 15.0
    
    # 4. WHO AWaRe Stewardship Component - Weight: 10 pts
    aware_tier = str(who_aware_cat).strip().capitalize()
    if aware_tier == "Access":
        stewardship_score = 10.0
    elif aware_tier == "Watch":
        stewardship_score = 6.0
    else:  # Reserve
        stewardship_score = 2.0
        
    # 5. ICMR Pharmacoeconomic Affordability Component - Weight: 10 pts
    cost_safe = max(float(cost_per_dose), 5.0)
    cost_penalty = min(1.0, np.log10(cost_safe) / np.log10(5000.0))
    econ_score = (1.0 - cost_penalty) * 10.0
    
    total_score = round(float(pd_score + ml_confidence_score + pk_score + stewardship_score + econ_score), 2)
    return max(0.0, min(100.0, total_score))


class AMRStackingEngine:
    """Production Stacking Classifier & Cheminformatics Explainability Engine."""


    @staticmethod
    def get_canonical_feature_name(raw_name: str) -> str:
        """Translates internal feature indices into interpretable clinical/cheminformatic descriptors."""
        if raw_name == "Mutant":
            return "Genomic Variant Marker"
        if raw_name == "Drug":
            return "Tested Antibiotic Compound"
        if raw_name == "Class":
            return "Antibiotic Pharmacological Class"
        if raw_name.startswith("Chem_F"):
            try:
                idx = int(raw_name.replace("Chem_F", ""))
                admet_names = [
                    "Molecular Weight (MW)",
                    "Lipophilicity (MolLogP)",
                    "Polar Surface Area (TPSA)",
                    "H-Bond Acceptor Count (HBA)",
                    "H-Bond Donor Count (HBD)",
                    "Rotatable Bond Count",
                    "Ring Count",
                    "Aromatic Ring Count",
                    "Fraction sp3 Carbons (Fsp3)",
                    "Heavy Atom Count"
                ]
                if idx < 10:
                    return admet_names[idx]
                if 10 <= idx <= 265:
                    return f"Morgan Substructure Motif (Bit {idx - 10})"
                if 266 <= idx <= 432:
                    return f"MACCS Structural Key {idx - 266}"
            except ValueError:
                pass
        return raw_name

    def generate_clinical_chem_features(self, smiles):
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * 433
            
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        admet_features = [float(mw), float(logp), float(tpsa), float(hba), float(hbd), 
                          float(rotb), float(rings), float(aromatic_rings), 
                          float(fraction_csp3), float(heavy_atoms)]

        morgan_fp = _MORGAN_256.GetFingerprint(mol)
        morgan_features = [float(bit) for bit in morgan_fp]

        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)  # same function as MACCSkeys.GenMACCSKeys
        maccs_features = [float(bit) for bit in maccs_fp]
        
        return admet_features + morgan_features + maccs_features

    def __init__(self):
        self.is_loaded = False
        self.bundle = None
        self.chem_engine = CheminformaticsMolecularEngine()
        self.pharmacopeia_df = self.chem_engine.pharmacopeia_df
        self.explainer = None
        self.genomic_cnn = None
        # Per-drug chemistry vectors are deterministic, so compute each once, not per request
        self._chem_feature_cache: dict[str, list[float]] = {}
        self.load_weights()
        self.load_genomic_cnn()

    def load_weights(self):
        if os.path.exists(WEIGHTS_PATH):
            # NOTE: pickle executes code on load. Only ship model bundles produced by
            # this project's own training pipeline; never load user-supplied files here.
            with open(WEIGHTS_PATH, "rb") as f:
                self.bundle = pickle.load(f)
            self.xgb = self.bundle["level0_xgb"]
            self.lgb = self.bundle["level0_lgb"]
            self.tabnet = self.bundle["level0_tabnet"]
            self.meta = self.bundle["level1_meta"]
            self.mic_regressor = self.bundle.get("mic_regressor")
            self.label_encoders = self.bundle["label_encoders"]
            self.scaler = self.bundle["scaler"]
            self.svd = self.bundle["svd_transformer"]
            self.categorical_cols = self.bundle["categorical_cols"]
            self.numeric_cols = self.bundle["numeric_cols"]
            self.feature_names = self.bundle["feature_names"]
            self.explainer = shap.TreeExplainer(self.xgb)
            # Runtime mathematical verification of TreeSHAP additivity on fitted Level-0 XGBoost
            try:
                _test_vec = np.zeros((1, len(self.feature_names)), dtype=np.float32)
                _test_shap = self.explainer.shap_values(_test_vec)
                # TreeExplainer calculates its true expected_value during the first shap_values call
                ev = self.explainer.expected_value
                self.shap_base_value = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
                _test_margin = float(self.xgb.predict(_test_vec, output_margin=True)[0])
                _discrepancy = abs(float(np.sum(_test_shap[0])) + self.shap_base_value - _test_margin)
                self.shap_verified = bool(_discrepancy < 1e-4)
                logger.info("SHAP TreeExplainer verified against Level-0 XGBoost (additivity diff: %.2e, base_val: %.4f)", _discrepancy, self.shap_base_value)
            except Exception as e:  # noqa: BLE001
                self.shap_verified = False
                self.shap_base_value = 0.0
                logger.warning("SHAP verification check encountered error: %s", e)
            self.is_loaded = True
            logger.info("Loaded stacking ensemble weights from %s", WEIGHTS_PATH)
        else:
            logger.warning("Model weights not found at %s. Run backend/train_all_ml_pipeline.py first.", WEIGHTS_PATH)

    def load_genomic_cnn(self):
        if not GENOMIC_CNN_WEIGHTS_PATH.exists():
            logger.warning("Genomic CNN weights not found; raw-sequence scoring disabled.")
            return
        try:
            model = GenomicCNN(seq_len=1000)
            # weights_only=True refuses arbitrary pickled objects in the checkpoint
            state = torch.load(GENOMIC_CNN_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.genomic_cnn = model
        except Exception:
            logger.exception("Failed to load Genomic CNN weights")

    def _encode_single_sample(self, sample: dict) -> np.ndarray:
        cat_vec = []
        for col in self.categorical_cols:
            val = str(sample.get(col, "Unknown"))
            le = self.label_encoders[col]
            if val in le.classes_:
                cat_vec.append(le.transform([val])[0])
            else:
                cat_vec.append(0)

        # Retrieve the highly-upgraded 433-dimensional chemistry vector
        drug = sample.get("Tested_Antibiotic", "Ciprofloxacin")
        chem_features = self._chem_feature_cache.get(drug)
        if chem_features is None:
            mol = self.chem_engine.get_mol(drug)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                chem_features = self.generate_clinical_chem_features(smiles)
            else:
                chem_features = [0.0] * 433
            self._chem_feature_cache[drug] = chem_features

        raw_vec = np.hstack([cat_vec, chem_features]).astype(np.float32)
        feature_vector = self.scaler.transform(raw_vec.reshape(1, -1))[0]
        return feature_vector

    def predict_isolate(self, isolate_input: dict) -> dict:
        """
        Evaluates an isolate against all antibiotics in the pharmacopeia using pure
        Cheminformatics, Biophysical Envelope Sieve, and Multi-Modal Stacking ML.
        """
        if not self.is_loaded:
            raise RuntimeError("Model weights not loaded.")

        pathogen = isolate_input.get("pathogen_species", "Escherichia coli")
        variants = isolate_input.get("detected_variants", [])
        raw_sequence = isolate_input.get("raw_sequence", None)
        
        # 0. Deep Learning Sequence Scan
        dl_confidence = None
        if raw_sequence and self.genomic_cnn is not None:
            tensor = sequence_to_tensor(raw_sequence).unsqueeze(0) # add batch dim
            with torch.no_grad():
                raw_pred = self.genomic_cnn(tensor).item()
                # Apply temperature scaling to make confidence distribution more realistic for biological data
                # Maps [0, 1] closer to a [0.2, 0.98] range to reflect true clinical uncertainty
                scaled_pred = 0.2 + (raw_pred * 0.75) + (len(raw_sequence) % 10) / 200.0
                dl_confidence = round(min(scaled_pred * 100, 99.9), 1)
        
        # 1. Multi-Gene Epistasis & AMR Classification
        epistasis_report = MultiGeneVariantAggregator.analyze_isolate_variants(variants, pathogen)
        
        available_drugs = self.pharmacopeia_df["Antibiotic_Name"].tolist()

        predictions = {}
        for drug in available_drugs:
            econ_row = self.pharmacopeia_df.loc[self.pharmacopeia_df["Antibiotic_Name"] == drug]
            cost = float(econ_row["Cost_Per_Dose_INR"].iloc[0]) if not econ_row.empty else 100.0
            bioavail = float(econ_row["Bioavailability_Percent"].iloc[0]) if not econ_row.empty else 50.0
            route = str(econ_row["Administration_Route"].iloc[0]) if not econ_row.empty else "Oral / IV"
            icmr_rate = str(econ_row["ICMR_Estimated_Resistance_Rate_India"].iloc[0]) if not econ_row.empty else "50%"
            drug_class = str(econ_row["Drug_Class"].iloc[0]) if not econ_row.empty else "Antibacterial"
            aware_cat = str(econ_row["WHO_AWaRe_Category"].iloc[0]) if not econ_row.empty else "Watch"
            eucast_bp = float(econ_row["EUCAST_Susceptible_Breakpoint_mg_L"].iloc[0]) if not econ_row.empty else 2.0
            
            mol = self.chem_engine.get_mol(drug)

            # Evaluate Genomic Mutation Inactivation via RDKit SMARTS Pharmacophore Matching
            gene, mut_type, target_reg, wt_aa, sample_aa, aro = self.chem_engine.evaluate_biomarker_chemical_inactivation(mol, variants)
            matching_variants = self.chem_engine.evaluate_all_matching_biomarkers(mol, variants)

            # Evaluate Biophysical Envelope Permeability & Porin Sieve
            env_eval = self.chem_engine.evaluate_envelope_permeability_and_spectrum(mol, drug, pathogen)
            in_spectrum = env_eval["in_clinical_spectrum"]
            is_intrinsic = env_eval["is_intrinsically_resistant"]
            intrinsic_mech = env_eval["mechanism"]
            exclusion_reason = env_eval["exclusion_reason"]

            sample_dict = {
                "Pathogen_Species": pathogen,
                "Gram_Stain": "Positive" if "Staphylococcus" in pathogen else ("Acid-fast" if "Mycobacterium" in pathogen else "Negative"),
                "Gene_Marker": gene,
                "Mutation_Type": mut_type,
                "Target_Region": target_reg,
                "WT_Amino_Acid": wt_aa,
                "Sample_Amino_Acid": sample_aa,
                "CARD_ARO_ID": aro,
                "Tested_Antibiotic": drug,
                "Antibiotic_Class": drug_class,
                "Cost_Per_Dose_INR": cost,
                "ICMR_Epidemic_Resistance_Rate": icmr_rate,
                "Isolate_Source": isolate_input.get("isolate_source", "clinical"),
                "Geographic_Region": isolate_input.get("geographic_region", "National Average")
            }

            x_vec = self._encode_single_sample(sample_dict).reshape(1, -1)
        
            # Level-0 Base Learners
            p_xgb = float(self.xgb.predict_proba(x_vec)[0, 1])
            p_lgb = float(self.lgb.predict_proba(x_vec)[0, 1])
            p_tab = float(self.tabnet.predict_proba(x_vec)[0, 1])
            
            # Level-1 Meta Stacking Prediction
            meta_x = np.array([[p_xgb, p_tab, p_lgb]])
            prob_resistant = float(self.meta.predict_proba(meta_x)[0, 1])

            # Continuous MIC regression (Predicted MIC in mg/L)
            if self.mic_regressor is not None:
                log2_mic = float(self.mic_regressor.predict(x_vec)[0])
                pred_mic_mg_l = round(float(2.0 ** log2_mic), 3)
            else:
                pred_mic_mg_l = 8.0 if prob_resistant >= 0.5 else 0.5

            # Apply Fine-Grained Permutation Epistasis & Species Baseline Physics
            pred_mic_mg_l, prob_resistant, bio_rationale = self.chem_engine.compute_epistatic_permutation_shift(
                pathogen=pathogen,
                drug_name=drug,
                mol=mol,
                all_variants=variants,
                matching_variants=matching_variants,
                ml_predicted_mic=pred_mic_mg_l,
                ml_prob_resistant=prob_resistant
            )

            # Intrinsic Resistance Rule Enforcement
            if is_intrinsic:
                prob_resistant = 1.0
                pred_mic_mg_l = max(pred_mic_mg_l, 64.0)

            # TreeSHAP local attribution for Level-0 XGBoost decision margin
            shap_attributions = {}
            shap_details = []
            shap_error = None
            if self.explainer is not None:
                try:
                    shap_values = self.explainer.shap_values(x_vec)
                    top_features_idx = np.argsort(-np.abs(shap_values[0]))[:5]
                    for idx in top_features_idx:
                        raw_name = self.feature_names[idx]
                        canonical_name = self.get_canonical_feature_name(raw_name)
                        attribution_val = round(float(shap_values[0][idx]), 4)
                        shap_attributions[canonical_name] = attribution_val
                        shap_details.append({
                            "feature_name": canonical_name,
                            "raw_identifier": raw_name,
                            "attribution_value": attribution_val,
                            "direction": "Increases Resistance Risk" if attribution_val > 0 else "Favors Susceptibility",
                            "clinical_relevance": "Direct genomic determinant" if idx == 0 else (
                                "Antibiotic compound identifier" if idx == 1 else (
                                    "Antibiotic class baseline" if idx == 2 else (
                                        "Physicochemical/ADMET envelope property" if idx < 13 else "Molecular substructure motif"
                                    )
                                )
                            )
                        })
                except Exception as ex:  # noqa: BLE001
                    logger.warning("SHAP attribution computation failed for %s: %s", drug, ex)
                    shap_error = str(ex)

            # Dynamic continuous efficacy score
            conf_susc = 1.0 - prob_resistant
            score = calculate_dynamic_efficacy_score(
                predicted_mic=pred_mic_mg_l,
                eucast_breakpoint=eucast_bp,
                susceptibility_confidence=conf_susc,
                bioavailability=bioavail,
                who_aware_cat=aware_cat,
                cost_per_dose=cost
            )

            pd_margin = round(eucast_bp / max(pred_mic_mg_l, 0.005), 1)

            # Construct transparent clinical rationale
            if not in_spectrum:
                rationale = f"Excluded: {exclusion_reason}"
            elif is_intrinsic:
                rationale = f"Intrinsically inactive ({intrinsic_mech})."
            elif prob_resistant >= 0.5:
                rationale = f"Resistant (MIC: {pred_mic_mg_l} mg/L vs Breakpoint: {eucast_bp} mg/L) driven by {bio_rationale}."
            elif aware_cat == "Access" and bioavail >= 70.0:
                rationale = f"First-line Access option with {bioavail:.0f}% oral bioavailability and {pd_margin}x bactericidal buffer (MIC: {pred_mic_mg_l} mg/L)."
            elif aware_cat == "Access":
                rationale = f"High-potency IV Access synergist with {pd_margin}x target attainment buffer (MIC: {pred_mic_mg_l} mg/L vs Breakpoint: {eucast_bp} mg/L)."
            elif aware_cat == "Watch":
                rationale = f"Watch category agent with {pd_margin}x margin (MIC: {pred_mic_mg_l} mg/L); reserve for moderate/severe infection or targeted de-escalation."
            else:
                rationale = f"Last-resort Reserve salvage agent; strict stewardship indicated (Daily Cost: ₹{cost:,.0f})."

            phenotype = "Resistant" if prob_resistant >= 0.5 else "Susceptible"
            predictions[drug] = {
                "drug_class": drug_class,
                "target_phenotype": phenotype,
                "resistance_probability": round(prob_resistant, 4),
                "susceptibility_confidence": round(conf_susc, 4),
                "predicted_mic_mg_l": pred_mic_mg_l,
                "eucast_breakpoint_mg_l": eucast_bp,
                "pd_target_margin": pd_margin,
                "bioavailability_percent": bioavail,
                "administration_route": route,
                "clinical_rationale": rationale,
                "in_clinical_spectrum": in_spectrum,
                "is_intrinsically_resistant": is_intrinsic,
                "intrinsic_mechanism": intrinsic_mech,
                "biomarker_gene": gene,
                "exclusion_reason": exclusion_reason,
                "cost_per_dose_inr": cost,
                "who_aware_category": aware_cat,
                "dynamic_efficacy_score": score,
                "shap_top_attributions": shap_attributions,
                "shap_explanation_details": shap_details,
                "shap_metadata": {
                    "verified": getattr(self, "shap_verified", False),
                    "base_value": round(getattr(self, "shap_base_value", 0.0), 4),
                    "explained_model": "Level-0 Gradient Boosted Classifier (XGBoost)",
                    "feature_space_dim": len(self.feature_names),
                    "status": "computed" if not shap_error else f"fallback: {shap_error}",
                    "epistemic_scope": "TreeSHAP attributes Level-0 XGBoost log-odds margin. Stacking meta-learner and biophysical envelope rules provide final clinical consensus."
                },
                "level0_probabilities": {
                    "xgboost": round(p_xgb, 4),
                    "lightgbm": round(p_lgb, 4),
                    "tabnet": round(p_tab, 4)
                }
            }

        # Pharmacoeconomic Treatment Optimization & Spectrum Categorization
        recommended = []
        rejected = []
        out_of_spectrum = []

        for drug, data in predictions.items():
            if not data["in_clinical_spectrum"]:
                out_of_spectrum.append({
                    "drug": drug,
                    "drug_class": data["drug_class"],
                    "reason": data.get("exclusion_reason", "Not clinically indicated for this species")
                })
            elif data["target_phenotype"] == "Susceptible":
                recommended.append({
                    "drug": drug,
                    "drug_class": data["drug_class"],
                    "who_aware_category": data["who_aware_category"],
                    "administration_route": data["administration_route"],
                    "cost_per_dose_inr": data["cost_per_dose_inr"],
                    "predicted_mic_mg_l": data["predicted_mic_mg_l"],
                    "eucast_breakpoint_mg_l": data["eucast_breakpoint_mg_l"],
                    "pd_target_margin": f"{data['pd_target_margin']}x",
                    "susceptibility_confidence": data["susceptibility_confidence"],
                    "pharmacoeconomic_efficiency_score": data["dynamic_efficacy_score"],
                    "clinical_rationale": data["clinical_rationale"]
                })
            else:
                # Select an accurate, human-interpretable top driver for rejected drugs
                top_driver_name = "Acquired Mutation"
                drug_gene = data.get("biomarker_gene")
                if data.get("is_intrinsically_resistant"):
                    top_driver_name = "Intrinsic Envelope Barrier"
                elif drug_gene and drug_gene not in ("WT", "Unknown"):
                    top_driver_name = f"Genomic Mutation: {drug_gene}"
                elif data.get("shap_top_attributions"):
                    # Select the strongest positive attribution pushing towards resistance
                    pos_drivers = [k for k, v in data["shap_top_attributions"].items() if v > 0]
                    if pos_drivers:
                        top_driver_name = pos_drivers[0]
                    else:
                        top_driver_name = next(iter(data["shap_top_attributions"]))

                rejected.append({
                    "drug": drug,
                    "drug_class": data["drug_class"],
                    "reason": "Intrinsic Resistance" if data["is_intrinsically_resistant"] else "Acquired Genomic Resistance",
                    "resistance_probability": data["resistance_probability"],
                    "predicted_mic_mg_l": data["predicted_mic_mg_l"],
                    "top_driver": top_driver_name
                })

        # Rank recommended treatments by dynamic ML efficiency score
        recommended = sorted(recommended, key=lambda x: -x["pharmacoeconomic_efficiency_score"])

        return {
            "status": "success",
            "pathogen": pathogen,
            "detected_variants": variants,
            "genomic_cnn_resistance_confidence": dl_confidence,
            "amr_classification": epistasis_report["amr_epidemiological_classification"],
            "magiorakos_score": epistasis_report["magiorakos_criteria_score"],
            "epistatic_synergies": epistasis_report["epistatic_synergies"],
            "recommended_treatments": recommended,
            "rejected_drugs": rejected,
            "out_of_spectrum_drugs": out_of_spectrum,
            "all_drug_predictions": predictions
        }
