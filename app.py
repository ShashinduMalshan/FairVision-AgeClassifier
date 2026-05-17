import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"
MODEL_PATH = "FairVision.pt"
AGE_GROUPS = ["0–2", "3–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70+"]

st.set_page_config(page_title="FairVision AI", page_icon="👁️", layout="wide")

# ── DARK THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── GLOBAL DARK BACKGROUND ── */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"],
section[data-testid="stSidebar"], .main {
  background: #080810 !important;
  color: #c8c7e0 !important;
  font-family: 'Space Grotesk', sans-serif !important;
}
.block-container { padding: 2.2rem 3rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── WORDMARK ── */
.fv-wordmark {
  font-family: 'Space Mono', monospace;
  font-size: 2.2rem; font-weight: 700;
  color: #e8e7f8; letter-spacing: -0.04em; margin-bottom: 4px;
}
.fv-wordmark em { font-style: normal; color: #7c6fff; }
.fv-tagline { font-size: 0.82rem; color: #5a5878; margin-bottom: 1rem; }
.fv-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 1.6rem; }
.pill { font-size: 0.64rem; font-family: 'Space Mono', monospace;
  letter-spacing: 0.08em; padding: 4px 11px; border-radius: 20px; border: 1px solid; }
.pill-indigo { background: #14123a; color: #9b93ff; border-color: #2e2870; }
.pill-emerald { background: #061a12; color: #3ecf8e; border-color: #0f4030; }
.pill-amber { background: #1a1208; color: #f0a840; border-color: #4a3010; }
.fv-hr { border: none; border-top: 1px solid #18182a; margin: 0 0 1.8rem; }

/* ── MODE TOGGLE BUTTONS (Bright when active, dim when not) ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div {
  display: flex !important; gap: 10px !important; flex-direction: row !important;
  margin-bottom: 4px !important;
}
div[data-testid="stRadio"] > div > label {
  display: flex !important; align-items: center; gap: 8px;
  padding: 11px 24px !important; border-radius: 10px !important; cursor: pointer;
  font-size: 0.84rem !important; font-weight: 600 !important; font-family: 'Space Grotesk', sans-serif !important;
  transition: all 0.18s ease;
  /* DIM state — unselected */
  background: #111122 !important;
  border: 1.5px solid #1e1e38 !important;
  color: #3a3860 !important;
}
/* BRIGHT state — selected */
div[data-testid="stRadio"] > div > label:has(input:checked) {
  background: #7c6fff !important;
  border-color: #9b8fff !important;
  color: #ffffff !important;
  box-shadow: 0 0 20px #7c6fff55 !important;
}
div[data-testid="stRadio"] input { display: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: #0d0d1e !important;
  border: 1.5px dashed #2a2850 !important;
  border-radius: 14px !important;
  padding: 1.8rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #3a3860 !important; font-size: 0.78rem !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: #1c1840 !important; color: #9b8fff !important;
  border: 1px solid #3a3270 !important; border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important;
}

/* ── CAMERA INPUT ── */
[data-testid="stCameraInput"] {
  background: #0d0d1e !important;
  border: 1.5px dashed #2a2850 !important;
  border-radius: 14px !important;
}
[data-testid="stCameraInput"] button {
  background: #1c1840 !important; color: #9b8fff !important;
  border: 1px solid #3a3270 !important; border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
}

/* ── IMAGE ── */
[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid #1e1e38 !important; }

/* ── SECTION LABELS ── */
.sec-label {
  font-family: 'Space Mono', monospace; font-size: 0.56rem;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #3a3860; border-bottom: 1px solid #14142a;
  padding-bottom: 8px; margin-bottom: 14px;
}

/* ── HERO RESULT CARD ── */
.hero-card {
  background: #0d0d22; border: 1px solid #1e1c42;
  border-radius: 16px; padding: 26px 28px; margin-bottom: 14px;
}
.hero-eyebrow {
  font-family: 'Space Mono', monospace; font-size: 0.58rem;
  letter-spacing: 0.2em; color: #4a4878; margin-bottom: 8px;
}
.hero-age {
  font-family: 'Space Mono', monospace; font-size: 3.5rem;
  font-weight: 700; color: #e8e7f8; line-height: 1; letter-spacing: -0.04em;
}
.hero-unit { font-size: 0.95rem; color: #7c6fff; margin-left: 8px; }
.conf-track { background: #1a1838; border-radius: 2px; height: 3px; margin-top: 16px; }
.conf-fill { background: #7c6fff; height: 3px; border-radius: 2px; }
.hero-conf {
  font-family: 'Space Mono', monospace; font-size: 0.74rem;
  color: #5a5878; margin-top: 8px;
}
.hero-conf strong { color: #b0a8ff; }

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
  background: #0d0d22 !important; border: 1px solid #1e1c42 !important;
  border-radius: 12px !important; padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {
  font-size: 0.6rem !important; letter-spacing: 0.15em !important;
  text-transform: uppercase !important; color: #4a4878 !important;
  font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 1.35rem !important; font-weight: 700 !important; color: #d8d7f0 !important;
}
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; }
[data-testid="stMetricDeltaIcon"] { display: none !important; }

/* ── BAR CHART ── */
[data-testid="stVegaLiteChart"] {
  background: #0d0d22 !important; border: 1px solid #1e1c42 !important;
  border-radius: 12px !important; padding: 14px !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div {
  background: #1a1838 !important; border-radius: 2px !important; height: 4px !important;
}
[data-testid="stProgressBar"] > div > div {
  background: #7c6fff !important; border-radius: 2px !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] p { color: #7c6fff !important; }

/* ── FAIRNESS NOTICE ── */
.fair-notice {
  background: #0c0c24; border: 1px solid #1e1c42; border-left: 3px solid #7c6fff;
  border-radius: 0 10px 10px 0; padding: 12px 16px;
  font-size: 0.75rem; color: #5a5878; line-height: 1.7; margin-top: 16px;
}
.fair-notice b { color: #9b8fff; }

/* ── EMPTY STATE ── */
.empty-state {
  text-align: center; padding: 4rem 0; color: #2a2848;
  font-family: 'Space Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
}

/* ── ALERTS / ERRORS ── */
[data-testid="stAlert"] {
  background: #180c0c !important; border: 1px solid #3a1818 !important;
  border-radius: 10px !important; color: #c87878 !important;
}
</style>
""", unsafe_allow_html=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_ftrs, num_classes))
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    model = FairVisionResNet(num_classes=9)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    with torch.no_grad():
        out = model(transform(image).unsqueeze(0))
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(probs, 0)
    return pred.item(), float(conf), probs


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fv-wordmark">Fair<em>Vision</em></div>
<div class="fv-tagline">Age classification · Bias-mitigated · ResNet-50</div>
<div class="fv-pills">
  <span class="pill pill-indigo">ResNet-50 backbone</span>
  <span class="pill pill-emerald">Bias mitigated</span>
  <span class="pill pill-amber">9-class output</span>
</div>
<div class="fv-hr"></div>
""", unsafe_allow_html=True)

# ── MODE TOGGLE ───────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    [" Upload Image", "  Live Camera"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("<br>", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:
    st.markdown('<div class="sec-label">Input</div>', unsafe_allow_html=True)
    image = None

    if "Upload" in mode:
        file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)
            w, h = image.size
            m1, m2 = st.columns(2)
            m1.metric("Width", f"{w}px")
            m2.metric("Height", f"{h}px")
    else:
        cam = st.camera_input("Capture", label_visibility="collapsed")
        if cam:
            image = Image.open(cam).convert("RGB")
            st.image(image, use_container_width=True)

    if image is None:
        st.markdown('<div class="empty-state">— awaiting input —</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="sec-label">Analysis</div>', unsafe_allow_html=True)

    if image is None:
        st.markdown('<div class="empty-state">— no prediction yet —</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running inference..."):
            idx, conf, probs = predict(image)

        label = AGE_GROUPS[idx]
        conf_pct = conf * 100

        st.markdown(f"""
        <div class="hero-card">
          <div class="hero-eyebrow">Predicted age group</div>
          <div class="hero-age">{label}<span class="hero-unit">yrs</span></div>
          <div class="conf-track"><div class="conf-fill" style="width:{int(conf_pct)}%"></div></div>
          <div class="hero-conf">Confidence: <strong>{conf_pct:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(conf))

        st.markdown('<div class="sec-label" style="margin-top:20px">Top candidates</div>', unsafe_allow_html=True)
        top3_val, top3_idx = torch.topk(probs, 3)
        c1, c2, c3 = st.columns(3)
        for rank, (col, val, i) in enumerate(zip([c1, c2, c3], top3_val, top3_idx)):
            col.metric(f"#{rank+1}", AGE_GROUPS[i.item()], f"{val.item()*100:.1f}%")

        st.markdown('<div class="sec-label" style="margin-top:20px">Probability distribution</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Age Group": AGE_GROUPS,
            "Probability (%)": [round(p.item() * 100, 1) for p in probs]
        })
        st.bar_chart(df.set_index("Age Group"), color="#7c6fff", height=200)

        st.markdown("""
        <div class="fair-notice">
          <b>Fairness audit</b> — Label smoothing and dropout regularisation reduce performance
          gaps across racial phenotypes. Predictions are probabilistic; do not use as sole
          determinants in sensitive contexts.
        </div>
        """, unsafe_allow_html=True)