# 🔍 How to Read This Code (VS Code Tour)

If you are a judge or developer opening this project in VS Code, here is how to "connect the dots" and trace the exact flow of data through our decoupled architecture.

### 📍 START HERE: The User Interface
1. Open **`frontend/app.py`**.
2. Scroll to **Line 66**: `uploaded_file = st.file_uploader(...)`. This is where the raw DNA FASTA string enters the system.
3. Scroll to **Line 79**: You will see a `requests.post()` sending the FASTA string directly to the Backend via the REST API endpoint `http://127.0.0.1:8000/api/v1/ingest/sequence`.

### 📍 DOT 2: The API Router
1. Now, open **`backend/main.py`**.
2. Scroll to **Line 33**: `@app.post("/api/v1/ingest/sequence")`. This is the exact route that receives the frontend's payload.
3. Notice how it passes the sequence to our PyTorch CNN module to extract the mutations (e.g., `blaNDM-1`).

### 📍 DOT 3: The Core Artificial Intelligence
1. Still in `backend/main.py`, look at **Line 55**: `@app.post("/api/v1/predict/isolate")`. This receives the extracted mutations and asks the AI what drugs will work.
2. Open **`backend/ml_engine.py`**. This is the heart of the project.
3. Look at **Line 269 (`predict_resistance`)**: This function receives the mutations.
4. Open **`backend/chemistry_engine.py`**. Notice how it uses the `RDKit` library to calculate the exact chemical descriptors (Molecular Weight, TPSA, LogP) for antibiotics.
5. Back in `ml_engine.py` (**Line 365**), the XGBoost **Stacking Ensemble** merges the Genomic markers from PyTorch with the Chemical descriptors from RDKit. It calculates the exact Clinical Efficacy Score and returns the JSON payload back to the frontend.

### 📍 DOT 4: Back to the User
1. Return to **`frontend/app.py`** at **Line 160**.
2. The UI receives the `recs` (Recommendations) and `rejects` (Contraindications) from the backend's JSON and renders the massive, color-coded clinical tables you see on the screen!

---
💡 **To Run the Code in VS Code:**
Just go to the "Run and Debug" tab on the left panel and click the green play button for **"🚀 Run AMrit Pipeline"**. We have pre-configured `.vscode/launch.json` to handle everything automatically!
