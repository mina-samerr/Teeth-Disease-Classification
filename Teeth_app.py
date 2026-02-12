import streamlit as st
import numpy as np
import time
import os
import json
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import gdown

# PAGE CONFIG

st.set_page_config(
    page_title="Teeth Disease Classifier",
    page_icon="🦷",
    layout="wide"
)



# GLOBAL STYLE (MEDICAL DARK THEME)


st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #000;
    color: white;
}

h1, h2, h3, h4, p, label {
    color: white !important;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.4em;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background-color: #3b82f6;
}

.card {
    background: #0f172a;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #2563eb;
}

hr {
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# NAVBAR


if "page" not in st.session_state:
    st.session_state.page = "Home"

nav1, nav2 = st.columns([4, 1])
with nav1:
    st.markdown("## 🦷 Teeth Disease Classifier")
with nav2:
    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("About"):
        st.session_state.page = "About"

st.markdown("<hr>", unsafe_allow_html=True)


# MODEL LOADING


@st.cache_resource
def load_teeth_model():
    file_id = '1-U9Uq1dGKWGQgefcKK8IO_HmBO7zl0Nc' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'vgg_teeth_model.h5'
    
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive... This may take a minute."):
            gdown.download(url, output, quiet=False)
            
    return load_model(output)

model = load_teeth_model()

class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
class_info = {
    "CaS": "Dental Caries",
    "CoS": "Calculus",
    "Gum": "Gingivitis",
    "MC": "Mouth Cancer",
    "OC": "Oral Cancer",
    "OLP": "Oral Lichen Planus",
    "OT": "Other"
}

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr / 255.0

# HOME PAGE

if st.session_state.page == "Home":

    # HERO
    st.markdown("""
    <div class="card">
        <h1>AI-Based Teeth Disease Detection</h1>
        <p>Using Deep Learning (VGG19) & Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)


    #  LEFT (UPLOAD)

    with left:
        st.markdown("### 📤 Upload Dental Image")
        uploaded_file = st.file_uploader(
            "Supported formats: JPG, PNG",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

    
    #  RIGHT (RESULT)

    with right:
        st.markdown("### 🩺 Diagnosis Result")

        if uploaded_file and st.button("🔍 Classify", use_container_width=True):
            st.toast("Analyzing image...")

            with st.spinner("Model is running..."):
                time.sleep(1.5)
                processed = preprocess_image(img)
                preds = model.predict(processed)[0]
                idx = np.argmax(preds)
                confidence = preds[idx]

            st.markdown(f"""
            <div class="card">
                <h3>Predicted Disease</h3>
                <h2 style="color:#60a5fa">{class_labels[idx]}</h2>
                <p><b>Description:</b> {class_info[class_labels[idx]]}</p>
                <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence))

            st.markdown("#### 📊 Class Probabilities")
            for i, label in enumerate(class_labels):
                st.write(f"{label} : {preds[i]*100:.2f}%")

        if not uploaded_file:
            st.info("Please upload an image to start.")

# 
# ABOUT PAGE
# 
else:
    st.markdown(
        '<div class="card">', unsafe_allow_html=True
    )

    st.markdown("## 🎓 About the Project")

    st.markdown(
        """
This final project presents an **AI-based system** that applies  
**Deep Learning and Computer Vision** techniques to automatically classify  
**dental diseases from medical images**.
        """
    )

    st.markdown("---")

    st.markdown("### 🔬 Technical Details")
    st.markdown(
        """
- **Model Architecture:** VGG19 (Transfer Learning)  
- **Deep Learning Frameworks:** TensorFlow & Keras  
- **Web Framework:** Streamlit  
- **Application Domain:** Medical AI Assistance  
        """
    )

    st.markdown("---")

    st.markdown("### 🎯 Project Objective")
    st.markdown(
        """
The main goal of this system is to assist **dentists, students, and researchers**
in the early detection and classification of dental diseases using
**Artificial Intelligence**.
        """
    )


    st.markdown(
        '</div>', unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🦷 Teeth Disease Classifier | Final Project | Mina")
