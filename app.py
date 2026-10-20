import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# ---------------------------------------------------------
# ML-PAM | Minimal black & red interface
# Keep these model artifacts beside app.py.
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML-PAM",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #090909;
        --panel: #111111;
        --panel-soft: #151515;
        --border: #292929;
        --text: #f4f4f4;
        --muted: #9a9a9a;
        --red: #e32636;
        --red-dark: #9f1724;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(227, 38, 54, .12), transparent 42%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3, .brand, .result-title { font-family: 'Space Grotesk', sans-serif !important; }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom: 2rem; }
    .brand-mark {
        width: 38px; height: 38px; display:flex; align-items:center; justify-content:center;
        background: var(--red); color: white; border-radius: 10px; font-size: 18px; font-weight: 700;
        box-shadow: 0 0 28px rgba(227,38,54,.18);
    }
    .brand { font-size: 1.35rem; letter-spacing: -.05em; font-weight: 700; color: #fff; }
    .brand span { color: var(--red); }
    .brand-sub { color: var(--muted); font-size: .75rem; margin-top: 2px; }
    .eyebrow { color: var(--red); text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size: clamp(2rem, 5vw, 3rem); line-height:1.08; letter-spacing:-.06em; font-weight:700; margin:.5rem 0 .7rem; color:#fff; }
    .hero-title span { color: var(--red); }
    .hero-copy { color: #a6a6a6; font-size: 1rem; line-height: 1.6; max-width: 600px; margin-bottom: 1.6rem; }
    .section-label { color:#f1f1f1; font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; margin: 1.25rem 0 .7rem; }
    div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
        color:#c7c7c7 !important; font-size:.82rem !important; font-weight:600 !important;
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {
        background:#101010 !important; border:1px solid #303030 !important; border-radius:9px !important;
    }
    input, textarea { color:#f5f5f5 !important; }
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: var(--red) !important; box-shadow:0 0 0 1px var(--red) !important;
    }
    div[data-testid="stButton"] > button {
        background:var(--red); color:#fff; border:1px solid var(--red); border-radius:9px;
        padding:.72rem 1rem; min-height:3rem; font-weight:700; transition:all .18s ease;
    }
    div[data-testid="stButton"] > button:hover { background:#bd1f2d; border-color:#bd1f2d; transform:translateY(-1px); }
    div[data-testid="stButton"] > button:focus { box-shadow:0 0 0 2px #090909, 0 0 0 4px var(--red); }
    .card {
        background:linear-gradient(145deg,#151515,#101010); border:1px solid var(--border);
        border-radius:14px; padding:1.1rem 1.2rem; margin: .65rem 0 1rem;
    }
    .card-label { color:var(--muted); font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }
    .result-title { font-size:1.5rem; font-weight:700; margin-top:.3rem; }
    .trusted { color:#f3f3f3; }
    .suspicious { color:#ff4757; }
    .confidence { color:#b6b6b6; font-size:.9rem; margin-top:.35rem; }
    .tiny-note { color:#777; font-size:.78rem; line-height:1.5; }
    hr { border-color:#252525 !important; margin:1.7rem 0 !important; }
    footer { visibility:hidden; }
    #MainMenu { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-row">
      <div class="brand-mark">M</div>
      <div>
        <div class="brand">ML<span>·</span>PAM</div>
        <div class="brand-sub">Access request intelligence</div>
      </div>
    </div>
    <div class="eyebrow">NLP · Request screening</div>
    <div class="hero-title">Make access decisions<br><span>with more context.</span></div>
    <div class="hero-copy">Review a privileged-access request and classify it as trusted or suspicious using your trained text and metadata model.</div>
    """,
    unsafe_allow_html=True,
)

# Artifact names expected from the training notebook.
ARTIFACTS = {
    "model": "model.pkl",
    "body": "body_vectorizer.pkl",
    "subject": "subject_vectorizer.pkl",
    "metadata": "metadata_preprocessor.pkl",
}

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(ARTIFACTS["model"]),
        joblib.load(ARTIFACTS["body"]),
        joblib.load(ARTIFACTS["subject"]),
        joblib.load(ARTIFACTS["metadata"]),
    )

def make_metadata_frame(preprocessor, department, request_type, urgency, amount):
    """Adapt the input row to the column names expected by the fitted preprocessor."""
    if hasattr(preprocessor, "feature_names_in_"):
        columns = list(preprocessor.feature_names_in_)
    else:
        columns = ["department", "request_type", "urgency", "amount"]
    values = {
        "department": department,
        "request_type": request_type,
        "urgency": urgency,
        "amount": amount,
    }
    # For expected columns not represented by this simple UI, use neutral defaults.
    row = {col: values.get(col, "unknown") for col in columns}
    return pd.DataFrame([row], columns=columns)

def transform_features(body, subject, metadata_row, body_vectorizer, subject_vectorizer, preprocessor):
    body_features = body_vectorizer.transform([body])
    subject_features = subject_vectorizer.transform([subject])
    metadata_features = preprocessor.transform(metadata_row)
    # Keep these multipliers aligned with the training notebook.
    return hstack(
        [body_features * 1.0, subject_features * 0.7, csr_matrix(metadata_features) * 1.5],
        format="csr",
    )

st.markdown('<div class="section-label">01 / Request details</div>', unsafe_allow_html=True)
subject = st.text_input("Email subject", placeholder="e.g. Request for production database access")
body = st.text_area(
    "Request message",
    placeholder="Paste the access request message here…",
    height=155,
)
left, right = st.columns(2)
with left:
    department = st.selectbox("Department", ["Engineering", "Finance", "HR", "IT", "Operations", "Sales", "Other"])
    request_type = st.selectbox("Access type", ["Privileged access", "Database access", "Admin access", "Application access", "Temporary access", "Other"])
with right:
    urgency = st.selectbox("Urgency", ["Low", "Normal", "High", "Critical"])
    amount = st.number_input("Requested access duration (hours)", min_value=1, max_value=8760, value=8, step=1)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run = st.button("Analyze request  →", use_container_width=True)

if run:
    if not subject.strip() or not body.strip():
        st.warning("Add both a subject and a request message before analyzing.")
    else:
        try:
            model, body_vec, subject_vec, metadata_prep = load_artifacts()
            metadata_row = make_metadata_frame(metadata_prep, department, request_type, urgency, amount)
            features = transform_features(body, subject, metadata_row, body_vec, subject_vec, metadata_prep)
            prediction = str(model.predict(features)[0]).lower()
            probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

            suspicious = "suspicious" in prediction
            result_class = "suspicious" if suspicious else "trusted"
            display_label = "Suspicious request" if suspicious else "Trusted request"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">02 / Analysis result</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card"><div class="card-label">Model classification</div>'
                f'<div class="result-title {result_class}">{display_label}</div>'
                f'<div class="confidence">Subject and message evaluated with request metadata.</div></div>',
                unsafe_allow_html=True,
            )

            if probabilities is not None:
                classes = [str(c).lower() for c in model.classes_]
                trusted_idx = next((i for i, c in enumerate(classes) if "trusted" in c), None)
                suspicious_idx = next((i for i, c in enumerate(classes) if "suspicious" in c), None)
                if trusted_idx is not None and suspicious_idx is not None:
                    trusted_prob = float(probabilities[trusted_idx])
                    suspicious_prob = float(probabilities[suspicious_idx])
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Trusted score", f"{trusted_prob:.1%}")
                    with c2:
                        st.metric("Suspicious score", f"{suspicious_prob:.1%}")
                    st.progress(max(0.0, min(1.0, suspicious_prob)), text="Suspicious score")
            if suspicious:
                st.error("Review recommended. Verify the requester, business justification, and requested privilege before approval.")
            else:
                st.success("Classified as trusted by the model. This result is a screening signal, not an approval decision.")
        except FileNotFoundError as exc:
            st.error(f"Model artifact not found: {exc.filename or exc}. Place all four .pkl files beside app.py.")
        except Exception as exc:
            st.error("The model could not process this request. Check that the artifacts and feature preprocessing match the training notebook.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="tiny-note">ML-PAM is a decision-support prototype. A classification is not proof of intent and should not replace identity verification, policy checks, or human review.</div>',
    unsafe_allow_html=True,
)