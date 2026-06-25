import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Taekwondo Pose Estimation")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Taekwondo Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    results = model.predict(
        source=temp_path,
        conf=0.25
    )

    annotated = results[0].plot()

    st.image(
        annotated,
        caption="Pose Estimation Result",
        use_container_width=True
    )