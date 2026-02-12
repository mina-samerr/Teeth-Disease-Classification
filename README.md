# 🦷 Teeth Disease Classification AI App
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An end-to-end Deep Learning web application designed to classify various dental diseases from medical images using Computer Vision.

## 🚀 Live Demo
You can try the application here: **[https://teeth-disease-classification-ynlzasgsu5vvorpnex9zvy.streamlit.app/]**

## 📝 Project Overview
Early detection of dental diseases can significantly improve treatment outcomes. This project leverages **Transfer Learning** with the **VGG19** architecture to provide an automated diagnostic tool for dentists and medical researchers.

### Key Features:
- **High Accuracy:** Optimized using pre-trained VGG19 weights.
- **Interactive UI:** A modern dark-themed medical dashboard.
- **Real-time Prediction:** Get instant results with confidence percentages.
- **Multi-class Classification:** Detects 7 different conditions.

## 📊 Dataset Classes
The model is trained to identify:
1. **CaS:** Dental Caries
2. **CoS:** Calculus
3. **Gum:** Gingivitis
4. **MC:** Mouth Cancer
5. **OC:** Oral Cancer
6. **OLP:** Oral Lichen Planus
7. **OT:** Others

## 🛠️ Technical Stack
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** VGG19
- **Web Framework:** Streamlit
- **Libraries:** NumPy, Pillow, gdown
