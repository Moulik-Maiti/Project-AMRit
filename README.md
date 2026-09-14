<div align="center">

# 🧬 AMrit: End-to-End AI Clinical Antibiogram Pipeline




[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

*A Next-Generation Genomic & Cheminformatics Platform for combating Antimicrobial Resistance (AMR).* 

</div>




AMrit (Antimicrobial Resistance Intelligence) is a robust, decoupled AI platform designed to take raw bacterial DNA sequences (FASTA) and output actionable clinical antibiograms. It determines exactly which antibiotics are safe to use, which are contraindicated (Resistant), and precisely *why* using cheminformatics.

---

## 🏗️ System Architecture & Backend Flow

The backend is built as a high-performance **FastAPI microservice** (`backend/main.py`), which orchestrates three distinct AI engines. 

### 1. The Backend Engines (`/backend/`)
* **`dl_genomics.py` (Genomic Extraction):** When a sequence is uploaded, this module runs a **PyTorch 1D-CNN** to scan the raw DNA sequence and extract specific clinical resistance markers (e.g., `blaNDM-1`, `mcr-1`, `gyrA_S83L`).
* **`chemistry_engine.py` (Cheminformatics):** Utilizes **RDKit** to map the extracted mutations against an antibiotic database. It generates precise chemical reasoning for why a drug fails (e.g., *beta-lactamase hydrolysis of pharmacophore ring* or *porin translocation blockage*).
* **`ml_engine.py` (Stacking Meta-Learner):** The core intelligence. It loads pre-trained weights (`/models/amr_ensemble_weights.pkl`) for an **XGBoost & LightGBM Stacking Ensemble**. It combines the genomic markers with the RDKit chemistry descriptors to calculate exact Pharmacodynamic MICs (Minimum Inhibitory Concentrations) and authentic Clinical Efficacy Scores.

### 2. The Total Pipeline Flow
1. **Input:** Raw FASTA DNA string is POSTed to the backend.
2. **Extraction:** PyTorch identifies the resistance loci.
3. **Synthesis:** The Stacking Ensemble pairs those loci with 14+ different antibiotics.
4. **Scoring:** The AI calculates the probability of resistance and the MIC value.
5. **Output:** The API returns a unified JSON containing Recommended Therapeutics (with Efficacy Scores) and Rejected Therapeutics (with RDKit Chemistry Logic).

---

## 🖥️ Frontend & API Connection

The user interface is an interactive dashboard built with **Streamlit** (`frontend/app.py`).

### How they connect:
* The backend runs a Uvicorn server on **`http://localhost:8000`**.
* The Streamlit frontend runs on **`http://localhost:8510`**.
* **Communication:** The frontend is completely decoupled. It communicates with the backend exclusively via REST API HTTP calls. 
  - `POST /api/v1/ingest/sequence`: Used to upload the FASTA file and trigger `dl_genomics.py` to extract mutations.
  - `POST /api/v1/predict/isolate`: Used to send the extracted mutations to `ml_engine.py` to get the final clinical antibiogram (Drugs, Scores, RDKit Logic).

---

## 🚀 How to Run the Application

This repository comes pre-packaged with a launcher script that handles both the frontend and backend simultaneously.

### Step 1: Install Dependencies
Ensure you have Python 3.10+ installed. Open your terminal in this directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Launch the Servers
Run the unified launcher script:
```bash
python run.py
```
This script will automatically:
1. Start the **FastAPI Backend Server** on port 8000.
2. Start the **Streamlit Frontend Dashboard** on port 8510.
3. Provide a local URL to click.

---

## 🧪 How to Demo the Pipeline

If you are presenting this project, follow this exact workflow:

1. Open **`http://localhost:8510`** in your browser.
2. We have provided custom sample genomes in the **`/demo_samples/`** folder. 
3. Drag and drop `Demo_E_coli_Superbug_Joint.fasta` into the Step 1 file uploader. 
4. The backend will instantly extract multiple mutations (`blaNDM-1` and `penA_mosaic`).
5. Click **"Synthesize Clinical Antibiogram"**. The app will dramatically cycle through the AI initialization steps before presenting the final XDR/MDR Clinical Report!
