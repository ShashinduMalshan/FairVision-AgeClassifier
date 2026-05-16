import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os


GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"

url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

gdown.download(url, "FairVision.pt", quiet=False)

# --- 1. MODEL ARCHITECTURE ---
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(FairVisionResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# --- 2. APP CONFIGURATION ---
st.set_page_config(
    page_title="FairVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f !important;
    color: #e0dff5 !important;
  }

  .stApp {
    background: #0a0a0f !important;
  }

  /* Header */
  .fv-header {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2.5rem;
  }
  .fv-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: #5f5fd4;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .fv-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0eff8;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .fv-title span {
    color: #7b7eff;
  }
  .fv-subtitle {
    font-size: 0.82rem;
    color: #6b6a80;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
  }

  /* Status chips */
  .chip-row {
    display: flex;
    gap: 8px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
  }
  .chip {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
  }
  .chip-blue  { background: #12123a; color: #7b7eff; border: 1px solid #2a2a5e; }
  .chip-green { background: #0d2420; color: #3ecf8e; border: 1px solid #1a4a3a; }
  .chip-amber { background: #2a1e08; color: #f5a623; border: 1px solid #4a3510; }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    background: #0e0e1a !important;
    border: 1px dashed #2a2a4a !important;
    border-radius: 4px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #5f5fd4 !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] {
    color: #6b6a80 !important;
    font-size: 0.8rem !important;
  }
  [data-testid="stFileUploaderDropzone"] button {
    background: #1a1a2e !important;
    color: #a0a0c0 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
  }

  /* Result card */
  .result-panel {
    background: #0e0e1a;
    border: 1px solid #1e1e32;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
  }
  .result-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5f5fa0;
    margin-bottom: 0.3rem;
  }
  .result-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #f0eff8;
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .result-value span {
    font-size: 1.2rem;
    color: #7b7eff;
    margin-left: 6px;
  }
  .conf-label {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5f5fa0;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
  }

  /* Progress bar override */
  [data-testid="stProgressBar"] > div {
    background: #1a1a2e !important;
    border-radius: 1px !important;
    height: 3px !important;
  }
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #5f5fd4, #7b7eff) !important;
    border-radius: 1px !important;
  }

  /* Bar chart styling */
  [data-testid="stVegaLiteChart"] {
    border: 1px solid #1e1e32;
    border-radius: 4px;
    padding: 1rem;
    background: #0e0e1a;
  }

  /* Metric overrides */
  [data-testid="stMetric"] {
    background: #0e0e1a;
    border: 1px solid #1e1e32;
    border-radius: 4px;
    padding: 1rem 1.2rem !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #5f5fa0 !important;
    font-family: 'DM Mono', monospace !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #f0eff8 !important;
  }

  /* Info box */
  .fair-box {
    background: #0d1520;
    border: 1px solid #1a2d4a;
    border-left: 3px solid #7b7eff;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    font-size: 0.78rem;
    color: #8888b0;
    line-height: 1.7;
  }
  .fair-box b {
    color: #a0a0d0;
  }

  /* Divider */
  hr {
    border: none;
    border-top: 1px solid #1a1a2a !important;
    margin: 2rem 0 !important;
  }

  /* Section headers */
  .section-head {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5f5fa0;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a2a;
  }

  /* Image display */
  [data-testid="stImage"] img {
    border-radius: 4px;
    border: 1px solid #1e1e32;
  }

  /* Spinner */
  [data-testid="stSpinner"] {
    color: #7b7eff !important;
  }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 0 !important; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# --- 3. HEADER ---
st.markdown("""
<div class="fv-header">
  <div class="fv-eyebrow">Responsible Face Analytics · v2.0</div>
  <div class="fv-title">Fair<span>Vision</span></div>
  <div class="fv-subtitle">Age classification with equitable performance across demographics</div>
  <div class="chip-row">
    <span class="chip chip-blue">ResNet-50 Backbone</span>
    <span class="chip chip-green">Bias Mitigated</span>
    <span class="chip chip-amber">9-Class Output</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = FairVisionResNet(num_classes=9)

    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "FairVision.pt")

    if not os.path.exists(model_path):
        with st.spinner("Downloading model weights..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False)

    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model 

model = load_model()

# --- 5. TWO-COLUMN LAYOUT ---
col_upload, col_results = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<div class="section-head">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a face image here",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

        # Image metadata
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        w, h = image.size
        m1.metric("Width", f"{w}px")
        m2.metric("Height", f"{h}px")

with col_results:
    st.markdown('<div class="section-head">Analysis</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="color: #3a3a5a; font-size: 0.8rem; padding: 3rem 0; text-align: center; letter-spacing: 0.08em;">
          — awaiting input —
        </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.error("Model unavailable. Check FairVision.pt path.")

    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)

        with st.spinner("Running inference..."):
            try:
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, pred = torch.max(probs, 0)

                age_groups = ["0–2", "3–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70+"]
                predicted_label = age_groups[pred.item()]
                confidence_pct = float(conf.item()) * 100

                # Main result
                st.markdown(f"""
                <div class="result-panel">
                  <div class="result-label">Predicted Age Group</div>
                  <div class="result-value">{predicted_label}<span>yrs</span></div>
                  <div class="conf-label">Confidence</div>
                  <div style="font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #7b7eff; margin-bottom: 0.6rem;">{confidence_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(conf.item()))

                # Distribution
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-head">Probability Distribution</div>', unsafe_allow_html=True)

                chart_data = {
                    "Age Group": age_groups,
                    "Probability": [float(p) * 100 for p in probs]
                }

                import pandas as pd
                df = pd.DataFrame(chart_data)
                df["Color"] = ["#7b7eff" if ag == predicted_label else "#2a2a4a" for ag in age_groups]

                st.bar_chart(
                    df.set_index("Age Group")["Probability"],
                    color="#7b7eff",
                    height=220
                )

                # Top-3
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-head">Top Candidates</div>', unsafe_allow_html=True)
                top3_vals, top3_idx = torch.topk(probs, 3)
                t1, t2, t3 = st.columns(3)
                cols_top = [t1, t2, t3]
                for i, (val, idx) in enumerate(zip(top3_vals, top3_idx)):
                    cols_top[i].metric(
                        f"#{i+1}",
                        age_groups[idx.item()],
                        f"{float(val)*100:.1f}%"
                    )

                # Fairness note
                st.markdown("""
                <div class="fair-box">
                  <b>Fairness Audit</b> — This model employs label smoothing and dropout regularisation
                  to reduce bias across racial phenotypes. Predictions are probabilistic and
                  should not be used as sole determinants in sensitive contexts.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")