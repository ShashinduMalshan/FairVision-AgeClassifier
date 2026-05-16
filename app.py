import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import pandas as pd

# =========================
# CONFIG
# =========================
GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"
MODEL_PATH = "FairVision.pt"

st.set_page_config(
    page_title="FairVision AI",
    page_icon="👁️",
    layout="wide"
)

# =========================
# MODEL
# =========================
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# =========================
# LOAD MODEL (FIXED)
# =========================
@st.cache_resource
def load_model():
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    model = FairVisionResNet(num_classes=9)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(cleaned, strict=False)
    model.eval()

    return model


model = load_model()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

age_groups = [
    "0–2", "3–9", "10–19", "20–29", "30–39",
    "40–49", "50–59", "60–69", "70+"
]


# =========================
# PREDICT
# =========================
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(probs, 0)

    return pred.item(), float(conf), probs


# =========================
# UI HEADER
# =========================
st.title("FairVision AI - Age Classification")

mode = st.radio("Mode", ["Upload Image", "Live Camera"])


# =========================
# MAIN LOGIC
# =========================
if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        idx, conf, probs = predict(image)
        label = age_groups[idx]

        # RESULT BOX
        st.markdown(f"""
        ## Prediction
        **{label}**
        Confidence: {conf*100:.2f}%
        """)

        # =========================
        # TOP 3 CANDIDATES (RESTORED)
        # =========================
        st.subheader("Top Candidates")
        top3_val, top3_idx = torch.topk(probs, 3)

        cols = st.columns(3)
        for i in range(3):
            cols[i].metric(
                f"Top {i+1}",
                age_groups[top3_idx[i].item()],
                f"{top3_val[i].item()*100:.2f}%"
            )

        # =========================
        # DISTRIBUTION
        # =========================
        st.subheader("Probability Distribution")

        df = pd.DataFrame({
            "Age Group": age_groups,
            "Probability": [p.item() * 100 for p in probs]
        })

        st.bar_chart(df.set_index("Age Group"))


# =========================
# LIVE CAMERA MODE
# =========================
elif mode == "Live Camera":

    cam = st.camera_input("Capture")

    if cam:
        image = Image.open(cam).convert("RGB")
        st.image(image, use_container_width=True)

        idx, conf, probs = predict(image)
        label = age_groups[idx]

        st.markdown(f"""
        ## Live Prediction
        **{label}**
        Confidence: {conf*100:.2f}%
        """)

        st.subheader("Top Candidates")

        top3_val, top3_idx = torch.topk(probs, 3)
        cols = st.columns(3)

        for i in range(3):
            cols[i].metric(
                f"Top {i+1}",
                age_groups[top3_idx[i].item()],
                f"{top3_val[i].item()*100:.2f}%"
            )