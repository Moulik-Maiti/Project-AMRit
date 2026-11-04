# AMRit

AMRit is a research prototype for exploring antimicrobial resistance data. It includes a
Streamlit interface, a FastAPI service, model artifacts, and reference datasets.

> **Limitations:** This software is not a diagnostic tool and does not provide treatment
> recommendations. Its outputs have not been established as a substitute for validated
> laboratory antimicrobial susceptibility testing (AST). Do not use them to make patient
> care decisions.

## What the application does

- The Streamlit interface can check uploaded sequence text for a small set of exact-match
  demo motifs defined in `frontend/app.py`. This is not general-purpose variant calling.
- The interface also sends the sequence to `POST /api/v1/ingest/sequence` and displays
  target loci returned by the backend. Those loci do not automatically populate the
  variant selections sent for prediction.
- The user selects a pathogen and variants. The interface sends those selections to
  `POST /api/v1/predict/isolate`.
- The backend returns recommended and rejected drug records and associated estimates.
  The interface displays those records; its resistance summary is a simple category
  based on how many drug classes the service rejects.
- `backend/chemistry_engine.py` uses RDKit to calculate chemical descriptors from the
  reference data. A descriptor or model output is not laboratory evidence of resistance.

## Run locally

Use Python 3.10 or later. From the repository root:

```bash
pip install -r requirements.txt
python run.py
```

The launcher starts the FastAPI backend on port `8000` and the Streamlit interface on
port `8510`.

## Configuration

| Variable | Component | Default | Purpose |
|---|---|---|---|
| `AMRIT_API_URL` | Frontend | `https://project-amrit.onrender.com` | Backend base URL; `run.py` sets the local backend URL |
| `PORT` / `HOST` | Backend | `8000` / `0.0.0.0` | Listen address when running `python backend/main.py` |
| `AMRIT_CORS_ORIGINS` | Backend | *(none)* | Comma-separated browser origins allowed to call the API |
| `AMRIT_ENABLE_DOCS` | Backend | `1` | Set to `0` to hide `/docs`, `/redoc`, and `/openapi.json` |
| `LOG_LEVEL` | Backend | `INFO` | Python logging level |
| `AMRIT_NUM_THREADS` | Backend | `1` | Thread count for numerical libraries during inference |
| `AMRIT_BACKEND_WAKE_TIMEOUT` | Frontend | `300` | Seconds to wait for the backend to become ready |
| `AMRIT_PREDICT_TIMEOUT` | Frontend | `180` | Read timeout for a prediction request, in seconds |

Dependency versions are pinned because the stored model artifacts may depend on specific
versions of scikit-learn and the model libraries. Pickle files can execute code when
loaded; only use model files from a trusted source.

## Sample data

Sample FASTA files are in `demo_samples/`. To try the interface, start the application
and upload one of these files. The frontend only matches the exact demo motifs configured
in its source; a missing match does not mean that the sample is susceptible. Select
variants manually to send them to the prediction endpoint.

See [CODE_TOUR_FOR_JUDGES.md](CODE_TOUR_FOR_JUDGES.md) for a brief guide to the request flow.
