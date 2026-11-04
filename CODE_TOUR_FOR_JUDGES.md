# Code tour

This guide follows the sequence and prediction requests through the application.

## 1. Streamlit interface

Start with `frontend/app.py`.

- The file uploader reads FASTA text and checks it against the small set of exact-match
  demo motifs in `DEMO_MARKER_MOTIFS`.
- `scan_sequence` also sends the sequence to the backend ingestion endpoint. The returned
  target loci are displayed separately; they are not used to select prediction variants.
- The variant checkboxes build the `detected_variants` list sent to the prediction endpoint.
- The interface displays the prediction response and adds a simple resistance category
  based on the number of rejected drug classes.

The local motif check is for the included demo data. It is not a general-purpose genomic
variant caller.

## 2. FastAPI routes

Open `backend/main.py` to see the service routes:

- `POST /api/v1/ingest/sequence` passes sequence text to the ingestion router and returns
  its result.
- `POST /api/v1/predict/isolate` passes the submitted isolate data to the prediction
  engine.
- `GET /api/v1/database/antibiotics` returns the antibiotic reference records.
- `GET /api/v1/health` reports service and model status.

## 3. Prediction and chemistry

`backend/ml_engine.py` contains the prediction engine. `backend/chemistry_engine.py`
contains chemistry-related calculations, including RDKit descriptor generation. Review
the implementation and model artifacts before interpreting any returned estimate; these
outputs are not a substitute for laboratory antimicrobial susceptibility testing.

## 4. Run the application

From the repository root, install the dependencies and start both services:

```bash
pip install -r requirements.txt
python run.py
```

The Streamlit interface is available at `http://localhost:8510`.
