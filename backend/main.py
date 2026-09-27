# ==================================================================
# 🧠 BACKEND API ROUTER
# This file listens on http://localhost:8000 for FastAPI requests.
# It receives data from frontend/app.py and routes it to ml_engine.py
# ==================================================================
"""
AMrit FastAPI Backend Service (Layer 6: Application & API Layer)
================================================================
Endpoints:
  - POST /api/v1/predict/isolate : Level-0 (XGBoost+TabNet+LGBM) + Level-1 Stacking + SHAP + Intrinsic Filter
  - POST /api/v1/ingest/sequence : Dynamic 6-Frame Pocket Extractor & QC
  - GET  /api/v1/database/antibiotics : 22-Drug Pharmacopeia with SMILES & ICMR Economics
  - GET  /api/v1/health : System status

Environment variables:
  - AMRIT_CORS_ORIGINS : comma-separated allowed browser origins (default: none; the
                         Streamlit frontend calls the API server-side and needs no CORS)
  - AMRIT_ENABLE_DOCS  : "1" to expose /docs and /openapi.json (default: enabled)
  - PORT               : listen port when run as a script (default: 8000)
"""

import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from ml_engine import AMRStackingEngine
from schemas import SequenceUploadRequest
from data_ingestion import UniversalIngestRouter, SingleIsolateInput

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("amrit.api")

docs_enabled = os.environ.get("AMRIT_ENABLE_DOCS", "1") == "1"

app = FastAPI(
    title="AMrit Clinical AMR Prediction & Intelligence API",
    description="Multi-Modal Genotype-to-Phenotype AI Engine powered by XGBoost, PyTorch TabNet, and RDKit",
    version="2.0.0",
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
)

cors_origins = [o.strip() for o in os.environ.get("AMRIT_CORS_ORIGINS", "").split(",") if o.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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
def health_check():
    return {
        "status": "healthy" if ml_engine.is_loaded else "degraded",
        "models_loaded": ml_engine.is_loaded,
        "genomic_cnn_loaded": ml_engine.genomic_cnn is not None,
        "supported_antibiotics_count": len(ml_engine.pharmacopeia_df),
        "engine_architecture": "Level-0 (XGBoost + TabNet + LightGBM) -> Level-1 (Logistic Regression Stacking Meta-Learner)",
        "chemistry_engine": "RDKit Cheminformatics + Biophysical Porin Sieve + SMARTS Pharmacophore Matcher"
    }


# Handlers below are plain `def` so FastAPI runs the CPU-bound model work in its
# threadpool instead of blocking the event loop.
@app.post("/api/v1/predict/isolate")
def predict_isolate_antibiogram(isolate: SingleIsolateInput):
    if not ml_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Prediction models are not loaded.")
    try:
        return ml_engine.predict_isolate(isolate.model_dump())
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/api/v1/ingest/sequence")
def ingest_genomic_sequence(req: SequenceUploadRequest):
    if not req.sequence_data.strip():
        raise HTTPException(status_code=400, detail="Sequence data cannot be empty.")
    try:
        results = ingest_router.ingest_payload(req.sequence_data)
    except Exception:
        logger.exception("Sequence ingestion failed")
        raise HTTPException(status_code=400, detail="Could not parse sequence data.")
    if results.get("status") == "error":
        raise HTTPException(status_code=400, detail=results.get("message", "Unrecognized input format."))
    return results


@app.get("/api/v1/database/antibiotics")
def get_antibiotics_database():
    if ml_engine.pharmacopeia_df.empty:
        raise HTTPException(status_code=503, detail="Pharmacopeia database unavailable")
    return {
        "status": "success",
        "total_drugs": len(ml_engine.pharmacopeia_df),
        "drugs": ml_engine.pharmacopeia_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        proxy_headers=True,
    )
