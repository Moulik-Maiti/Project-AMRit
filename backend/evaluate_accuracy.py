import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ENSEMBLE_WEIGHTS_PATH
from dl_genomics import GenomicCNN, sequence_to_tensor


def evaluate_cnn():
    print("--- EVALUATING 1D-CNN GENOMIC FEATURE EXTRACTOR ---")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "genomic_cnn_weights.pth")
    if not os.path.exists(model_path):
        print("CNN weights not found!")
        return None
        
    model = GenomicCNN(seq_len=1000)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    np.random.seed(42)  # reproducible validation set
    ndm1 = "ATGGAATTGCCCAATATTATGCACCCGGTCGCGAAGCTGAGC"
    parc = "GGCGTATTCGACCTGTAT"
    
    y_true = []
    y_pred_probs = []
    y_pred = []
    
    print("Testing on 500 clinical/synthetic validation sequences...")
    for i in range(500):
        is_resistant = np.random.rand() > 0.5
        seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=1000))
        if is_resistant:
            motif = ndm1 if np.random.rand() > 0.5 else parc
            insert_pos = np.random.randint(0, 1000 - len(motif))
            seq = seq[:insert_pos] + motif + seq[insert_pos + len(motif):]
            
        tensor = sequence_to_tensor(seq, 1000).unsqueeze(0)
        with torch.no_grad():
            prob = model(tensor).item()
            
        y_true.append(1 if is_resistant else 0)
        y_pred_probs.append(prob)
        y_pred.append(1 if prob >= 0.5 else 0)
        
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    print(f"CNN Accuracy: {acc*100:.2f}%")
    print(f"CNN AUC: {auc:.4f}")
    return {"Accuracy": acc, "AUC": auc}

def evaluate_ensemble():
    """
    Smoke-test the production stacking ensemble through AMRStackingEngine, exactly as the API
    calls it, and measure real per-isolate latency.

    Accuracy is NOT reported: that needs a labelled held-out antibiogram set, which is not
    shipped with this repo. Point this at one before quoting accuracy numbers.
    """
    print("\n--- EVALUATING STACKING META-LEARNER PHENOTYPE PREDICTION ---")
    if not os.path.exists(ENSEMBLE_WEIGHTS_PATH):
        print("Ensemble weights not found!")
        return None

    from data_ingestion import VALID_PATHOGENS
    from ml_engine import AMRStackingEngine

    engine = AMRStackingEngine()
    variant_sets = [[], ["gyrA_S83L"], ["blaNDM-1", "ompK36_porin_loss"], ["rpoB_S450L", "katG_S315T"], ["mecA", "vanA"]]

    latencies, failures, drug_calls = [], 0, 0
    for pathogen in VALID_PATHOGENS:
        for variants in variant_sets:
            start = time.perf_counter()
            try:
                result = engine.predict_isolate({"pathogen_species": pathogen, "detected_variants": variants})
            except Exception as e:  # noqa: BLE001 - report every failure and keep checking the rest
                failures += 1
                print(f"FAILED {pathogen} {variants}: {e.__class__.__name__}: {e}")
                continue
            latencies.append(time.perf_counter() - start)
            drug_calls += len(result.get("recommended_treatments", [])) + len(result.get("rejected_drugs", []))

    total = len(VALID_PATHOGENS) * len(variant_sets)
    print("\n[ENSEMBLE INFERENCE CHECK]")
    print(f"Isolates evaluated: {total - failures}/{total} succeeded ({drug_calls} drug predictions)")
    if latencies:
        print(f"Latency per isolate: median {np.median(latencies) * 1000:.0f} ms, max {max(latencies) * 1000:.0f} ms")
    print("Accuracy: not computed - requires a labelled held-out antibiogram dataset (none bundled).")
    return {"succeeded": total - failures, "total": total, "median_latency_s": float(np.median(latencies)) if latencies else None}


if __name__ == "__main__":
    evaluate_cnn()
    evaluate_ensemble()
