# ==================================================================n# 🧠 BACKEND API ROUTERn# This file listens on http://localhost:8000 for FastAPI requests.n# It receives data from frontend/app.py and routes it to ml_engine.pyn# ==================================================================n
"""
AMrit FastAPI Backend Service (Layer 6: Application & API Layer)
================================================================
Endpoints:
  - POST /api/v1/predict/isolate : Level-0 (XGBoost+TabNet+LGBM) + Level-1 Stacking + SHAP + Intrinsic Filter
  - POST /api/v1/ingest/sequence : Dynamic 6-Frame Pocket Extractor & QC
  - GET  /api/v1/database/antibiotics : 22-Drug Pharmacopeia with SMILES & ICMR Economics
  - GET  /api/v1/health : System status
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from ml_engine import AMRStackingEngine
from schemas import SequenceUploadRequest
from data_ingestion import UniversalIngestRouter, SingleIsolateInput

app = FastAPI(
    title="AMrit Clinical AMR Prediction & Intelligence API",
    description="Multi-Modal Genotype-to-Phenotype AI Engine powered by XGBoost, PyTorch TabNet, and RDKit",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
ml_engine = AMRStackingEngine()
ingest_router = UniversalIngestRouter()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AMrit Clinical AMR Platform",
        "message": "FastAPI backend is running"
    }
    
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ml_engine.is_loaded,
        "supported_antibiotics_count": len(ml_engine.pharmacopeia_df) if not ml_engine.pharmacopeia_df.empty else 22,
        "engine_architecture": "Level-0 (XGBoost + TabNet + LightGBM) -> Level-1 (Logistic Regression Stacking Meta-Learner)",
        "chemistry_engine": "RDKit Cheminformatics + Biophysical Porin Sieve + SMARTS Pharmacophore Matcher"
    }

@app.post("/api/v1/predict/isolate")
async def predict_isolate_antibiogram(isolate: SingleIsolateInput):
    try:
        results = ml_engine.predict_isolate(isolate.model_dump())
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest/sequence")
async def ingest_genomic_sequence(req: SequenceUploadRequest):
    if not req.sequence_data or req.sequence_data.strip() == "":
        raise HTTPException(status_code=400, detail="Sequence data cannot be empty.")
    try:
        results = ingest_router.ingest_payload(req.sequence_data)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/database/antibiotics")
async def get_antibiotics_database():
    if ml_engine.pharmacopeia_df.empty:
        return {"status": "error", "message": "Pharmacopeia database unavailable"}
    return {
        "status": "success",
        "total_drugs": len(ml_engine.pharmacopeia_df),
        "drugs": ml_engine.pharmacopeia_df.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
