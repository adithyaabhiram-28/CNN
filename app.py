# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("road_damage_cnn.h5")
    return model

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return encoder

model = load_model()
label_encoder = load_encoder()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
# 🚧 AI-Based Road Damage Detection System

### Smart City Infrastructure Monitoring using CNN
""")

st.divider()

# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------
st.subheader("📘 About the Project")

st.markdown("""
Road infrastructure plays a critical role in transportation safety and economic development.
Manual road inspection is expensive, time-consuming, and often inefficient.

### Why Road Monitoring is Important
- Prevent accidents caused by damaged roads.
- Reduce maintenance costs through early detection.
- Improve public safety and transportation efficiency.
- Enable smart city infrastructure management.

### Role of CNN in Computer Vision
Convolutional Neural Networks (CNNs) automatically learn visual patterns from images.
They can identify:
- Potholes
- Cracks
- Surface damage
- Road deterioration

CNNs provide fast and accurate road condition assessment using image analysis.

### Practical Industry Applications
- Smart City Monitoring
- Highway Maintenance Systems
- Municipal Road Inspection
- Autonomous Vehicles
- Infrastructure Asset Management
""")

st.divider()

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------------------------------------------------
# SEVERITY MAPPING
# ---------------------------------------------------
def get_severity(label):

    label = label.lower()

    if "pothole" in label:
        return "High"

    elif "crack" in label:
        return "Medium"

    else:
        return "Low"

# ---------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------
def get_recommendation(severity):

    if severity == "High":
        return (
            "Immediate maintenance recommended.",
            "⚠️ High-risk road condition detected."
        )

    elif severity == "Medium":
        return (
            "Schedule repair work soon.",
            "⚠️ Moderate damage may worsen over time."
        )

    else:
        return (
            "Routine monitoring recommended.",
            "✅ Road condition appears relatively safe."
        )

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.divider()

    # SECTION 4
    st.subheader("🖼 Uploaded Image Preview")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(
            image,
            caption="Uploaded Road Image",
            use_container_width=True
        )

    # Prediction
    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)

    predicted_index = np.argmax(prediction)

    class_name = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(np.max(prediction) * 100)

    severity = get_severity(class_name)

    recommendation, warning = get_recommendation(severity)

    # ---------------------------------------------------
    # SECTION 5
    # ---------------------------------------------------
    with col2:

        st.subheader("🔍 Prediction Results")

        st.success(f"Prediction: {class_name}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        if severity == "High":
            st.error(f"Severity: {severity}")

        elif severity == "Medium":
            st.warning(f"Severity: {severity}")

        else:
            st.info(f"Severity: {severity}")

    st.divider()

    # ---------------------------------------------------
    # SECTION 6
    # VISUALIZATION
    # ---------------------------------------------------
    st.subheader("📊 Visualization Area")

    class_labels = list(label_encoder.classes_)

    probs = prediction[0] * 100

    df = pd.DataFrame({
        "Damage Type": class_labels,
        "Confidence (%)": probs
    })

    col3, col4 = st.columns(2)

    with col3:

        fig1 = px.bar(
            df,
            x="Damage Type",
            y="Confidence (%)",
            title="Class Confidence Graph"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col4:

        fig2 = px.pie(
            df,
            values="Confidence (%)",
            names="Damage Type",
            title="Probability Distribution Chart"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    # ---------------------------------------------------
    # SECTION 7
    # RECOMMENDATIONS
    # ---------------------------------------------------
    st.subheader("🛠 Recommendations")

    st.info(recommendation)

    st.warning(warning)

    if severity == "High":
        st.error(
            "Road repair should be prioritized immediately to prevent accidents."
        )

    elif severity == "Medium":
        st.warning(
            "Periodic inspection and repair scheduling recommended."
        )

    else:
        st.success(
            "No urgent maintenance action required."
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.caption(
    "AI-Based Road Damage Detection System | Smart City Infrastructure Monitoring using CNN"
)
