import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import gdown

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Taekwondo Pose Estimation",
    page_icon="🥋",
    layout="wide"
)

MODEL_PATH = "best.pt"
FILE_ID = "1qMxA677fUY8ZBN1S3ZtqKANgRVqilnq_"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading trained model..."):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

    return YOLO(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🥋 Taekwondo Pose Estimation")

st.write(
    "Upload a Taekwondo image to detect body keypoints using YOLO Pose."
)

# --------------------------------------------------
# OVERALL METRICS
# --------------------------------------------------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

# Replace with your actual results

col1.metric("mAP@50", "96.4%")
col2.metric("mAP@50-95", "89.2%")
col3.metric("Precision", "94.8%")
col4.metric("Recall", "95.7%")

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.25,
    0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        results = model.predict(
            source=tmp.name,
            conf=confidence
        )

    predicted_image = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Predicted Image")
        st.image(
            predicted_image,
            use_container_width=True
        )
