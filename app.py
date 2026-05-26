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
# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
IMG_SIZE = (128, 128)

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)

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
    color:white;
    padding:10px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
}

.about-box{
    background:#161b22;
    padding:20px;
    border-radius:15px;
    border:1px solid #30363d;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_damage_cnn.h5")

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl","rb") as f:
        return pickle.load(f)

model = load_model()
encoder = load_encoder()

# --------------------------------------------------
# AUTO IMAGE SIZE
# --------------------------------------------------
INPUT_SIZE = (
    model.input_shape[1],
    model.input_shape[2]
)

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
def preprocess(img):

    img = img.convert("RGB")
    img = img.resize(INPUT_SIZE)

    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,axis=0)

    return arr

# --------------------------------------------------
# SEVERITY
# --------------------------------------------------
def get_severity(label):

    label = label.lower()

    if "pothole" in label:
        return "High"

    elif "crack" in label:
        return "Medium"

    else:
        return "Low"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🚧 AI-Based Road Damage Detection System</h1>
<h4>Smart City Infrastructure Monitoring using CNN</h4>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
with st.expander("📘 About Project", expanded=True):

    st.markdown("""
### Why Road Monitoring Matters

Road damage affects public safety, transportation efficiency,
vehicle maintenance costs, and smart-city development.

### CNN in Computer Vision

Convolutional Neural Networks automatically learn visual
patterns from road images and identify:

- Potholes
- Cracks
- Surface Deterioration
- Structural Damage

### Industry Applications

✅ Smart Cities

✅ Highway Monitoring

✅ Municipal Road Inspection

✅ Autonomous Vehicles

✅ Infrastructure Asset Management
""")

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.subheader("📤 Upload Road Image")

uploaded = st.file_uploader(
    "Drag & Drop or Browse Image",
    type=["jpg","jpeg","png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded:

    image = Image.open(uploaded)

    col1,col2 = st.columns([1.2,1])

    # IMAGE PREVIEW
    with col1:

        st.subheader("🖼 Uploaded Road Image")

        st.image(
            image,
            use_container_width=True
        )

    # PREDICT
    pred = model.predict(
        preprocess(image),
        verbose=0
    )

    idx = np.argmax(pred)

    label = encoder.inverse_transform([idx])[0]

    confidence = float(np.max(pred)*100)

    severity = get_severity(label)

    # RESULTS
    with col2:

        st.subheader("🔍 Prediction Results")

        c1,c2 = st.columns(2)

        with c1:
            st.metric(
                "Damage Type",
                label
            )

        with c2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.markdown("### Severity")

        if severity=="High":
            st.markdown(
                '<div class="high">HIGH RISK</div>',
                unsafe_allow_html=True
            )

        elif severity=="Medium":
            st.markdown(
                '<div class="medium">MEDIUM RISK</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                '<div class="low">LOW RISK</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    st.subheader("📊 Prediction Analytics")

    classes = list(encoder.classes_)

    df = pd.DataFrame({
        "Class": classes,
        "Confidence": pred[0]*100
    })

    col3,col4 = st.columns(2)

    with col3:

        fig = px.bar(
            df,
            x="Class",
            y="Confidence",
            title="Class Confidence Graph",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig2 = px.pie(
            df,
            names="Class",
            values="Confidence",
            title="Probability Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------
    st.subheader("🛠 Maintenance Recommendation")

    if severity=="High":

        st.error("""
🚨 Immediate maintenance recommended.

High-risk road condition detected.

Road repair should be prioritized immediately.
""")

    elif severity=="Medium":

        st.warning("""
⚠ Schedule maintenance soon.

Moderate damage may worsen if ignored.
""")

    else:

        st.success("""
✅ Road condition appears stable.

Routine monitoring recommended.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.caption(
    "AI-Based Road Damage Detection System • Smart City Infrastructure Monitoring using CNN"
)
