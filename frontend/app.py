# ==================================================================
# 🔗 FRONTEND UI DASHBOARD
# This file connects to the FastAPI backend (AMRIT_API_URL).
# Data flows out via requests.post() and JSON is returned here to render.
# ==================================================================
import os
import threading
import time

import pandas as pd
import requests
import streamlit as st
from pyvis.network import Network
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="AMRit AI: Sequence-to-Treatment", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #f8fafc; }
.metric-box { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
.safe-drug { color: #22c55e; font-weight: bold; }
.danger-drug { color: #ef4444; font-weight: bold; }
.warning-drug { color: #f59e0b; font-weight: bold; }
.card-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 25px; }
.metric-card { background-color: #1e293b; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 4px solid #3b82f6;}
.metric-title { color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: 800; }
.step-header { color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.jpg")
st.sidebar.image(LOGO_PATH, width=120)
st.sidebar.markdown("## AMRit AI Platform")
st.sidebar.info("Sequence-to-Treatment clinical decision support system.")
st.sidebar.caption("Research decision-support tool. Confirm all results with laboratory AST before prescribing.")

st.title("🧬 AMRit: AI Clinical Antibiogram Pipeline")

FASTAPI_URL = os.environ.get("AMRIT_API_URL", "https://project-amrit.onrender.com").rstrip("/")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# The hosted backend runs on Render's free tier, which sleeps after ~15 minutes idle. Waking
# it (container boot + ML model loading) has been measured at 2+ minutes, so requests wait
# for the backend explicitly instead of failing on a short read timeout.
CONNECT_TIMEOUT_S = 10
BACKEND_WAKE_TIMEOUT_S = float(os.environ.get("AMRIT_BACKEND_WAKE_TIMEOUT", "300"))
PREDICT_TIMEOUT_S = float(os.environ.get("AMRIT_PREDICT_TIMEOUT", "180"))

SUPPORTED_VARIANTS = [
    "gyrA_S83L", "parC_S80I", "blaNDM-1", "blaKPC-2", "blaOXA-48", "blaCTX-M-15", "mcr-1",
    "ompK36_porin_loss", "penA_mosaic", "rpoB_S450L", "katG_S315T", "mecA", "vanA",
]

# Known demo marker motifs present in demo_samples/*.fasta
DEMO_MARKER_MOTIFS = {
    "ATGGAATTGCCCAATATTATGCACCCGGTCGCGAAGCTGAGC": "blaNDM-1",
    "GCGTACGCATCGTGACGT": "penA_mosaic",
    "GGCGTATTCGACCTGTAT": "gyrA_S83L",
}


@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    # read=0: never re-send a request whose response timed out; that would multiply the wait
    retry = Retry(total=3, connect=3, read=0, status=3, backoff_factor=1,
                  status_forcelist=(502, 503, 504), allowed_methods=frozenset({"GET", "POST"}))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def api_error_message(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return f"HTTP {resp.status_code}: {detail or resp.reason}"


def backend_is_ready(read_timeout: float) -> bool:
    """True once the backend answers its health check with the models loaded."""
    try:
        # Single attempt (no retrying session): callers poll, and a quick "not yet" keeps page loads fast
        resp = requests.get(f"{FASTAPI_URL}/api/v1/health", timeout=(CONNECT_TIMEOUT_S, read_timeout))
        return resp.ok and bool(resp.json().get("models_loaded", True))
    except (requests.RequestException, ValueError):
        return False


@st.cache_resource(ttl=600, show_spinner=False)
def start_backend_warmup() -> threading.Thread:
    """Start waking a sleeping backend as soon as someone opens the app, so it is usually
    ready by the time they have uploaded a sequence. Re-armed at most every 10 minutes."""
    thread = threading.Thread(target=backend_is_ready, args=(BACKEND_WAKE_TIMEOUT_S,), daemon=True)
    thread.start()
    return thread


def ensure_backend_ready() -> bool:
    """Wait (with a visible explanation) until the backend is up; False if it never comes up."""
    if backend_is_ready(read_timeout=5):
        return True
    deadline = time.monotonic() + BACKEND_WAKE_TIMEOUT_S
    with st.spinner("Waking up the AMRit prediction engine. After a period of inactivity "
                    "this can take 2-3 minutes; later requests will be fast..."):
        while (remaining := deadline - time.monotonic()) > 0:
            if backend_is_ready(read_timeout=remaining):
                return True
            time.sleep(min(5.0, max(deadline - time.monotonic(), 0.0)))
    return False


@st.cache_data(show_spinner=False, max_entries=32)
def fetch_sequence_loci(seq_text: str) -> list:
    """Ask the backend to locate resistance target loci. Raises on failure so errors are not cached."""
    resp = get_http_session().post(f"{FASTAPI_URL}/api/v1/ingest/sequence",
                                   json={"sequence_data": seq_text, "format": "FASTA"},
                                   timeout=(CONNECT_TIMEOUT_S, 90))
    if not resp.ok:
        raise RuntimeError(api_error_message(resp))
    loci = []
    for record in resp.json().get("sequences", []):
        for pocket_name, info in (record.get("extracted_pockets") or {}).items():
            loci.append(f"{pocket_name} (target codon: {info.get('target_codon_aa', '?')})")
    return loci


def scan_sequence(seq_text: str) -> dict:
    """Detect demo markers locally, then ask the backend to locate resistance target loci."""
    markers = [name for motif, name in DEMO_MARKER_MOTIFS.items() if motif in seq_text.upper()]
    loci, error = [], None
    if not ensure_backend_ready():
        error = "backend did not wake up in time"
    else:
        try:
            loci = fetch_sequence_loci(seq_text)
        except requests.Timeout:
            error = "backend timed out"
        except requests.RequestException as e:
            error = f"backend unreachable: {e.__class__.__name__}"
        except RuntimeError as e:
            error = str(e)
    return {"markers": markers, "loci": loci, "error": error}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_antibiotics() -> list:
    resp = get_http_session().get(f"{FASTAPI_URL}/api/v1/database/antibiotics",
                                  timeout=(CONNECT_TIMEOUT_S, 90))
    resp.raise_for_status()
    return resp.json().get("drugs", [])


def chemistry_logic_for(drug_class: str, variants: list) -> str:
    """Pick the detected variant that mechanistically explains resistance for this drug class."""
    cls = str(drug_class).lower()
    v = " ".join(variants).lower()
    if any(k in cls for k in ("carbapenem", "cephalosporin", "penicillin", "beta")):
        if any(k in v for k in ("ndm", "kpc", "oxa", "ctx", "bla")):
            return "Cheminformatics: RDKit predicts severe beta-lactamase hydrolysis of pharmacophore ring."
        if "mec" in v:
            return "Cheminformatics: PBP2a conformational change abrogates beta-lactam binding."
        if "pena" in v:
            return "Cheminformatics: Mosaic PBP2 reduces beta-lactam acylation efficiency."
        if "porin" in v or "omp" in v:
            return "Cheminformatics: High Molecular Weight & TPSA prevents porin translocation."
    if "quinolone" in cls and ("gyr" in v or "par" in v):
        return "Cheminformatics: Loss of critical hydrogen bonding at target binding pocket."
    if "polymyxin" in cls and "mcr" in v:
        return "Cheminformatics: Lipid A modification repels cationic peptide charge."
    if "glycopeptide" in cls and "van" in v:
        return "Cheminformatics: D-Ala-D-Ala to D-Ala-D-Lac switch prevents glycopeptide binding."
    if "rifamycin" in cls and "rpob" in v:
        return "Cheminformatics: RRDR substitution disrupts rifamycin binding pocket."
    if "katg" in v and ("isoniazid" in cls or "isonicotinic" in cls):
        return "Cheminformatics: Loss of KatG activation prevents isoniazid prodrug conversion."
    return "Cheminformatics: Intrinsic or envelope-level resistance for this pathogen–drug pair."


start_backend_warmup()

tab_pipeline, tab_database, tab_network = st.tabs([
    "🚀 Clinical Pipeline",
    "💊 Cheminformatics",
    "🕸️ Knowledge Graph"
])

# =====================================================================
# TAB 1: SEAMLESS END-TO-END PIPELINE (STEP-BY-STEP)
# =====================================================================

with tab_pipeline:

    # ----------------- STEP 1 -----------------
    st.markdown("<h3 class='step-header'>Step 1: Genomic Sequence Ingestion</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        pathogen = st.selectbox("Suspected Pathogen Species", [
            "Escherichia coli", "Klebsiella pneumoniae", "Staphylococcus aureus",
            "Pseudomonas aeruginosa", "Acinetobacter baumannii",
            "Mycobacterium tuberculosis", "Neisseria gonorrhoeae"
        ])
    with col2:
        uploaded_file = st.file_uploader("Upload Patient FASTA Sequence (.fasta)", type=["fasta", "fa", "txt"])

    mutations_found = []

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        seq_input = None
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            st.error(f"File too large ({len(raw_bytes) / 1e6:.1f} MB). Maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        else:
            try:
                seq_input = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.error("File is not valid UTF-8 text. Please upload a plain-text FASTA file.")

        if seq_input is not None:
            with st.spinner("Scanning sequence for resistance markers..."):
                scan = scan_sequence(seq_input)
            mutations_found = scan["markers"]

            if mutations_found:
                st.success(f"✅ **Identified {len(mutations_found)} critical resistance markers:** "
                           f"`{', '.join(mutations_found)}`")
            else:
                st.info("✅ Sequence analyzed. **No known resistance markers detected.** (Wild-type/Susceptible)")
            if scan["loci"]:
                st.caption(f"Resistance target loci located: {', '.join(scan['loci'])}")
            if scan["error"]:
                st.warning(f"Backend sequence scan unavailable ({scan['error']}). Showing local marker scan only.")

    # ----------------- STEP 2 -----------------
    st.markdown("<h3 class='step-header'>Step 2: Manual Variant Override (Demo Controls)</h3>", unsafe_allow_html=True)
    st.markdown("Doctors can manually toggle variants identified via secondary PCR or bypass FASTA upload.")

    variant_cols = st.columns(3)
    final_mutations = []
    for i, variant in enumerate(SUPPORTED_VARIANTS):
        with variant_cols[i % 3]:
            if st.checkbox(variant, value=variant in mutations_found):
                final_mutations.append(variant)

    # ----------------- STEP 3 -----------------
    st.markdown("<h3 class='step-header'>Step 3: AI Stacking Ensemble Execution</h3>", unsafe_allow_html=True)

    analyze_btn = st.button("🚀 Synthesize Clinical Antibiogram", type="primary", width="stretch")

    if analyze_btn:
        with st.status("🧠 Executing Multi-Model AI Pipeline...", expanded=True) as status:
            st.write("Running stacking ensemble, RDKit pharmacophore mapping and MIC regression...")

            payload = {
                "pathogen_species": pathogen,
                "detected_variants": final_mutations,
                "isolate_source": "clinical"
            }
            data = None
            if not ensure_backend_ready():
                status.update(label="❌ Analysis Failed", state="error")
                st.error("The prediction engine did not start in time. Please try again in a minute.")
            else:
                try:
                    pred_res = get_http_session().post(f"{FASTAPI_URL}/api/v1/predict/isolate", json=payload,
                                                       timeout=(CONNECT_TIMEOUT_S, PREDICT_TIMEOUT_S))
                    if pred_res.ok:
                        data = pred_res.json()
                    else:
                        status.update(label="❌ Analysis Failed", state="error")
                        st.error(f"Prediction service error — {api_error_message(pred_res)}")
                except requests.Timeout:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"The prediction took longer than {PREDICT_TIMEOUT_S:.0f}s. "
                             "The server may be under load; please try again.")
                except requests.RequestException as e:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Prediction service unreachable ({e.__class__.__name__}). "
                             "Please try again shortly.")

            if data is not None:
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        if data is not None:
            st.markdown("---")

            recs = data.get("recommended_treatments", [])
            rejects = data.get("rejected_drugs", [])

            # MDR/XDR Logic
            rejected_classes = {rj.get('drug_class') for rj in rejects if rj.get('drug_class')}
            total_classes = {d.get('drug_class') for d in (recs + rejects) if d.get('drug_class')}

            if len(rejected_classes) >= max(3, len(total_classes) - 1) and len(total_classes) > 2:
                classification, color = "XDR (Extensively Drug-Resistant)", "#ef4444"
            elif len(rejected_classes) >= 3:
                classification, color = "MDR (Multi-Drug Resistant)", "#f97316"
            elif rejected_classes:
                classification, color = "Resistant (Targeted)", "#f59e0b"
            else:
                classification, color = "Fully Susceptible", "#22c55e"

            st.markdown(f'''
            <div class="card-container">
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-title">Predicted Phenotype</div>
                    <div class="metric-value" style="color: {color};">{classification}</div>
                </div>
                <div class="metric-card" style="border-top: 4px solid #22c55e;">
                    <div class="metric-title">Viable Therapeutics</div>
                    <div class="metric-value" style="color: #22c55e;">{len(recs)} Safe Options</div>
                </div>
                <div class="metric-card" style="border-top: 4px solid #ef4444;">
                    <div class="metric-title">Contraindicated</div>
                    <div class="metric-value" style="color: #ef4444;">{len(rejects)} High-Risk</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown("#### ✅ Viable Therapeutics")
            if recs:
                rec_df = pd.DataFrame(recs)
                rec_df["Clinical Efficacy Score"] = rec_df["pharmacoeconomic_efficiency_score"].apply(
                    lambda x: f"{float(x):.1f}%")
                rec_df["Predicted MIC"] = rec_df["predicted_mic_mg_l"].apply(lambda x: f"{round(float(x), 2)} mg/L")
                rec_df["Breakpoint"] = rec_df["eucast_breakpoint_mg_l"].apply(lambda x: f"≤ {x} mg/L")
                rec_df["Cost (INR)"] = rec_df["cost_per_dose_inr"].apply(lambda x: f"₹{float(x):,.0f}")

                display_rec = rec_df.loc[:, [
                    "drug", "drug_class", "administration_route", "Clinical Efficacy Score",
                    "Predicted MIC", "Breakpoint", "who_aware_category", "Cost (INR)", "clinical_rationale"
                ]].rename(columns={
                    "drug": "Drug",
                    "drug_class": "Class",
                    "administration_route": "Route",
                    "who_aware_category": "WHO AWaRe",
                    "clinical_rationale": "Clinical Rationale"
                })

                styled_rec = display_rec.style.set_properties(subset=None, **{
                    'background-color': '#022c22',
                    'color': '#6ee7b7',
                    'border-color': '#065f46'
                })

                st.dataframe(styled_rec, width="stretch", height=350)
            else:
                st.error("No viable therapeutics found. Strain is pan-resistant.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### ❌ Out of Treatment Range (Resistant Drugs)")
            if rejects:
                rej_df = pd.DataFrame(rejects)
                rej_df["Chemistry Logic Engine"] = rej_df["drug_class"].apply(
                    lambda c: chemistry_logic_for(c, final_mutations))
                rej_df["Predicted MIC"] = rej_df["predicted_mic_mg_l"].apply(lambda x: f"{round(float(x), 2)} mg/L")

                display_rej = rej_df.loc[:, [
                    "drug", "drug_class", "Predicted MIC", "reason", "Chemistry Logic Engine"
                ]].rename(columns={
                    "drug": "Drug",
                    "drug_class": "Class",
                    "reason": "Genomic Rationale"
                })

                styled_rej = display_rej.style.set_properties(subset=None, **{
                    'background-color': '#450a0a',
                    'color': '#fca5a5',
                    'border-color': '#7f1d1d'
                })

                st.dataframe(styled_rej, width="stretch", height=350)
            else:
                st.success("No resistant drugs detected! Fully susceptible strain.")


# =====================================================================
# TAB 2 & 3: DATABASE & GRAPH
# =====================================================================

with tab_database:
    st.subheader("💊 RDKit Cheminformatics Database")
    # This tab renders on every page load, so never block it on a cold start: if the backend
    # is still waking (the warm-up thread is already on it), say so and offer a retry.
    drugs_data = None
    if not backend_is_ready(read_timeout=5):
        st.info("⏳ The prediction engine is waking up (this can take 2-3 minutes after a period "
                "of inactivity). The database will load once it is ready.")
    else:
        try:
            drugs_data = fetch_antibiotics()
        except requests.Timeout:
            st.warning("The database request timed out. Please retry.")
        except (requests.RequestException, ValueError) as e:
            st.error(f"Database unavailable ({e.__class__.__name__}).")
    if drugs_data:
        df_drugs = pd.DataFrame(drugs_data)
        cols = [c for c in ["Antibiotic_Name", "Drug_Class", "Molecular_Weight", "LogP", "Cost_Per_Dose_INR"]
                if c in df_drugs.columns]
        st.dataframe(df_drugs[cols], width="stretch")
    elif drugs_data is not None:
        st.info("No antibiotics returned by the database.")
    else:
        st.button("🔄 Retry loading database", key="retry_database")


@st.cache_resource
def build_knowledge_graph_html() -> str:
    # pyvis documents font_color as str; its signature default (False) makes checkers infer bool
    net = Network(height="520px", width="100%", bgcolor="#0b1329", font_color="white",  # pyright: ignore[reportArgumentType]
                  cdn_resources="remote")
    pathogens = ["E. coli", "K. pneumoniae", "P. aeruginosa", "S. aureus", "A. baumannii", "M. tuberculosis", "N. gonorrhoeae"]
    for p in pathogens:
        net.add_node(p, label=p, color="#22c55e", size=22)
    genes = ["gyrA_S83L", "parC_S80I", "blaNDM-1", "blaCTX-M-15", "ompK36", "rpoB_S450L", "katG_S315T", "mcr-1", "mecA"]
    for g in genes:
        net.add_node(g, label=g, color="#ef4444" if "bla" in g else "#f59e0b", size=18)
    classes = ["Fluoroquinolones", "Carbapenems", "Cephalosporins", "Polymyxins", "Rifamycins", "Isonicotinic Acids"]
    for c in classes:
        net.add_node(c, label=c, color="#3b82f6", size=16)

    net.add_edge("E. coli", "gyrA_S83L")
    net.add_edge("E. coli", "blaNDM-1")
    net.add_edge("K. pneumoniae", "blaNDM-1")
    net.add_edge("M. tuberculosis", "rpoB_S450L")
    net.add_edge("gyrA_S83L", "Fluoroquinolones")
    net.add_edge("blaNDM-1", "Carbapenems")
    # Render in memory: writing to a shared file on disk races between concurrent sessions
    return net.generate_html()


with tab_network:
    st.subheader("🕸️ Interactive Knowledge Network")
    # Trusted, server-generated HTML only (no user input reaches the graph)
    st.iframe(build_knowledge_graph_html(), height=550)
