import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import gdown

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Taekwondo Pose Estimation",
    page_icon="🥋",
    layout="wide"
)

# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = "best.pt"
FILE_ID = "1qMxA677fUY8ZBN1S3ZtqKANgRVqilnq_"

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading trained YOLO Pose model..."):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🥋 Taekwondo Pose Estimation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Statistics",
        "Evaluation Metrics",
        "Pose Estimation Demo"
    ]
)

# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("🥋 Taekwondo Pose Estimation System")

    st.markdown("""
    ### Overview

    This system applies Human Pose Estimation using YOLO Pose
    for Taekwondo movement analysis.

    The model was trained using annotated Taekwondo forms:

    - Form 1
    - Form 5
    - Form 7

    The objective is to detect and visualize human body
    keypoints for posture analysis and technique assessment.
    """)

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    col1.info("Model: YOLO Pose")
    col2.info("Task: Human Pose Estimation")
    col3.info("Application: Taekwondo Form Analysis")

# ============================================================
# EVALUATION METRICS
# ============================================================

elif page == "Evaluation Metrics":

    st.title("📈 Model Evaluation")

    st.info(
        "Replace the values below with your actual validation results."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("mAP@50", "0.00")
    col2.metric("mAP@50-95", "0.00")
    col3.metric("Precision", "0.00")
    col4.metric("Recall", "0.00")

    st.divider()

    st.subheader("Training Curves")

    if os.path.exists("assets/results.png"):
        st.image(
            "assets/results.png",
            use_container_width=True
        )
    else:
        st.warning(
            "Upload results.png to assets folder."
        )

    st.divider()

    st.subheader("Confusion Matrix")

    if os.path.exists("assets/confusion_matrix.png"):
        st.image(
            "assets/confusion_matrix.png",
            use_container_width=True
        )
    else:
        st.warning(
            "Upload confusion_matrix.png to assets folder."
        )

    st.divider()

    st.subheader("Validation Predictions")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists("assets/val_batch0_pred.jpg"):
            st.image(
                "assets/val_batch0_pred.jpg",
                use_container_width=True
            )

    with col2:
        if os.path.exists("assets/val_batch1_pred.jpg"):
            st.image(
                "assets/val_batch1_pred.jpg",
                use_container_width=True
            )

# ============================================================
# POSE ESTIMATION DEMO
# ============================================================

elif page == "Pose Estimation Demo":

    st.title("🎯 Pose Estimation Demo")

    uploaded_file = st.file_uploader(
        "Upload a Taekwondo Image",
        type=["jpg", "jpeg", "png"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                image,
                use_container_width=True
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            image.save(tmp.name)

            results = model.predict(
                source=tmp.name,
                conf=confidence
            )

        annotated_image = results[0].plot()

        with col2:
            st.subheader("Pose Estimation Result")
            st.image(
                annotated_image,
                use_container_width=True
            )

        st.divider()

        st.subheader("Keypoint Information")

        try:

            keypoints = results[0].keypoints

            if keypoints is not None:

                xy = keypoints.xy.cpu().numpy()
                conf = keypoints.conf.cpu().numpy()

                joint_names = [
                    "Nose",
                    "Left Eye",
                    "Right Eye",
                    "Left Ear",
                    "Right Ear",
                    "Left Shoulder",
                    "Right Shoulder",
                    "Left Elbow",
                    "Right Elbow",
                    "Left Wrist",
                    "Right Wrist",
                    "Left Hip",
                    "Right Hip",
                    "Left Knee",
                    "Right Knee",
                    "Left Ankle",
                    "Right Ankle"
                ]

                if len(xy) > 0:

                    table = []

                    for i in range(min(len(joint_names), len(xy[0]))):

                        table.append(
                            {
                                "Joint": joint_names[i],
                                "X": round(float(xy[0][i][0]), 2),
                                "Y": round(float(xy[0][i][1]), 2),
                                "Confidence": round(
                                    float(conf[0][i]),
                                    3
                                )
                            }
                        )

                    df = pd.DataFrame(table)

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to display keypoint data: {e}"
            )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Taekwondo Pose Estimation using YOLO Pose"
)
