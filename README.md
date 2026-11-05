# AMRit

AMRit is a research prototype for exploring antimicrobial resistance (AMR) data. It pairs a
Streamlit interface with a FastAPI backend that combines machine-learning estimates,
sequence-processing utilities, chemistry descriptors, and rule-based checks.

> **Research use only.** AMRit is not a diagnostic tool and does not provide treatment
> recommendations. Its outputs have not been clinically validated and must not be used to
> make patient-care decisions or replace validated laboratory antimicrobial susceptibility
> testing (AST).

## What the application does

The user selects a pathogen and resistance markers in the interface. AMRit sends those
selections to the backend, which evaluates the supported antibiotics and returns estimates
alongside chemistry and rule-based context. The interface also accepts sequence files for
ingestion and displays loci identified by the backend.

The sequence upload and prediction steps are separate: loci returned from ingestion do not
automatically become the variants used in a prediction. The interface's local motif check
only recognizes a few exact demo sequences; it is not a general-purpose variant caller.

## Technology

- **Interface:** Streamlit
- **API:** FastAPI and Pydantic
- **Tabular ML:** XGBoost, LightGBM, scikit-learn histogram gradient boosting, and logistic
  regression stacking
- **Sequence ML:** PyTorch 1D convolutional neural network (CNN)
- **Sequence and chemical processing:** Biopython and RDKit
- **Interpretability:** SHAP attribution for the XGBoost classifier

The API combines model estimates with additional chemistry and rule-based processing.
Consequently, a returned estimate is not necessarily a direct output of one trained model.

## Repository map

| Path | Purpose |
|---|---|
| `frontend/app.py` | Streamlit user interface |
| `backend/main.py` | FastAPI application and HTTP routes |
| `backend/ml_engine.py` | Loads model artifacts and assembles isolate predictions |
| `backend/data_ingestion.py` | Input validation, sequence parsing, marker handling, and related rules |
| `backend/chemistry_engine.py` | RDKit chemistry calculations and associated rule-based processing |
| `backend/dl_genomics.py` | CNN architecture, sequence encoding, and synthetic CNN training |
| `backend/train_all_ml_pipeline.py` | Experimental tabular and CNN training entry point |
| `backend/train_clinical_sih_final.py` | Alternate experimental training pipeline |
| `backend/train_local_real_data.py` | Alternate data-download and training pipeline |
| `databases/` | Reference chemistry, antibiotic, and mechanism data |
| `demo_samples/` | Example FASTA files for trying the interface |
| `models/` | Serialized model artifacts loaded by the application |
| `CODE_TOUR_FOR_JUDGES.md` | Short guide to the request flow |

## Run locally

Use Python 3.10 or later. From the repository root, create and activate a virtual
environment, then install the pinned dependencies.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

The launcher starts both services:

- Streamlit: <http://localhost:8510>
- FastAPI: <http://127.0.0.1:8000>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Health status: <http://127.0.0.1:8000/api/v1/health>

The first startup can take time while the backend loads its model artifacts. The health
route reports whether the main model bundle was loaded and whether the genomic CNN is
available.

To run just the API from the repository root:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Stop the combined launcher with `Ctrl+C`.

## API overview

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Reports service and model-loading status |
| `GET` | `/api/v1/database/antibiotics` | Returns antibiotic reference records |
| `POST` | `/api/v1/ingest/sequence` | Parses submitted sequence text and returns ingestion results |
| `POST` | `/api/v1/predict/isolate` | Returns estimates for an isolate and supported antibiotics |

Example isolate prediction request:

```json
{
  "pathogen_species": "Escherichia coli",
  "detected_variants": ["blaNDM-1"]
}
```

Example sequence-ingestion request:

```json
{
  "sequence_data": ">demo\nATGGAATTGCCCAATATTATGCACCCGGTCGCGAAGCTGAGC",
  "format": "FASTA"
}
```

Use the API documentation at `/docs` for the current request and response schemas.
Supported species and input limits are defined in `backend/data_ingestion.py` and
`backend/schemas.py`.

## Model-training workflow and limitations

The repository contains several training scripts; they are experimental and do not all
implement the same feature pipeline. The default entry point is:

```bash
python -m backend.train_all_ml_pipeline
```

At a high level, that script:

1. Scans the bundled FASTA examples for resistance pockets and falls back to a small
   hard-coded marker/motif set if it finds none.
2. Calls `train_synthetic` to build random DNA sequences, insert selected motifs into
   positive examples, and train a 1D CNN using Adam and binary cross-entropy.
3. Creates 3,000 synthetic antibiotic/marker rows. The resistance labels and MIC values
   are assigned by programmed marker-and-drug-class rules, with additional random
   background resistance.
4. Encodes marker, drug, and drug-class categories; adds chemistry descriptors and a
   subset of Morgan fingerprint bits; and standardizes the tabular feature matrix.
5. Fits XGBoost, LightGBM, and scikit-learn `HistGradientBoostingClassifier` base models.
   The training script calls the third learner “TabNet,” but it is not a TabNet model.
6. Fits logistic regression to combine the base models' probabilities and trains a
   separate XGBoost regressor on log₂-transformed MIC values.

**Important: retraining is not currently a verified end-to-end workflow.** The default
training script does not use a held-out evaluation set, and its stacking model is trained
on in-sample base-model predictions. Its generated training data is synthetic, not a
patient antibiogram dataset. In addition, the bundle fields and feature layout written by
this script do not match what `backend/ml_engine.py` expects at inference. Running it can
overwrite files under `models/`; do not use its output for the application without first
resolving that mismatch and validating the complete train-to-inference path.

The alternate training scripts also generate synthetic labels or have their own distinct
feature pipelines. A script name or comments describing data as “clinical” do not establish
that a model was trained or validated on clinical outcome data.

The CNN evaluation in `backend/evaluate_accuracy.py` uses generated sequence examples.
The ensemble evaluation is a smoke test; the script explicitly does not report ensemble
accuracy because the repository does not include a labeled, held-out antibiogram dataset.
Do not present those checks as clinical performance metrics.

## Data and interpretation

- The FASTA files in `demo_samples/` are for demonstrating the workflow. A motif match or
  lack of a match is not a validated resistance call.
- Sequence ingestion returns candidate loci; users currently select variants separately
  before requesting a prediction.
- RDKit features describe molecular structure. They do not prove that a drug will work
  against an isolate.
- Rule-based intrinsic-resistance and epistasis logic contributes to outputs in addition
  to the learned models. Inspect the implementation and reference data before interpreting
  a result.
- Do not upload identifiable or sensitive patient data to a local or hosted instance.

## Configuration

| Variable | Component | Default | Purpose |
|---|---|---|---|
| `AMRIT_API_URL` | Frontend | `https://project-amrit.onrender.com` | Backend base URL; `run.py` sets it to the local API |
| `AMRIT_BACKEND_HOST` | Launcher | `127.0.0.1` | Backend bind host used by `run.py` |
| `AMRIT_BACKEND_PORT` | Launcher | `8000` | Backend port used by `run.py` |
| `AMRIT_FRONTEND_PORT` | Launcher | `8510` | Streamlit port used by `run.py` |
| `HOST` / `PORT` | API script | `0.0.0.0` / `8000` | Listen host and port when starting `backend.main` directly |
| `AMRIT_CORS_ORIGINS` | API | *(none)* | Comma-separated browser origins permitted by CORS |
| `AMRIT_ENABLE_DOCS` | API | `1` | Set to `0` to disable `/docs`, `/redoc`, and `/openapi.json` |
| `LOG_LEVEL` | API | `INFO` | Python logging level |
| `AMRIT_NUM_THREADS` | API | `1` | Numerical-library thread count during inference |
| `AMRIT_BACKEND_WAKE_TIMEOUT` | Frontend | `300` | Seconds to wait for the backend to become ready |
| `AMRIT_PREDICT_TIMEOUT` | Frontend | `180` | Prediction request read timeout in seconds |

## Validation and contribution

There is no automated test suite in the repository at present. For a local smoke check:

1. Start the application with `python run.py`.
2. Open `/api/v1/health` and confirm the expected model status.
3. Open `/docs` and try the sequence-ingestion route with a non-sensitive demo sample.
4. In the UI, upload a file from `demo_samples/`, select the pathogen and variants, and
   submit a prediction. Remember that ingestion results do not auto-select prediction
   variants.

Before changing model preprocessing, feature order, or model dependencies, check
compatibility with the stored artifacts. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request.

## Deployment

The repository's Vercel configuration points to `backend.main:app`, so that deployment
serves the API only. The Streamlit frontend must be deployed separately and configured with
`AMRIT_API_URL` set to the API's base URL. Review `pyproject.toml` and the target provider's
current Python and dependency support before deploying.

The model bundle is a pickle file. Pickle can execute code while loading; only load model
artifacts from a trusted source. The pinned dependencies are important for artifact
compatibility; changing them should include a model-loading check and, where required,
retraining and validation.
