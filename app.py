import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64

# Get the absolute path to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_base64_encoded_image(image_path):
    # Convert relative path to absolute path
    abs_image_path = os.path.join(SCRIPT_DIR, image_path)
    with open(abs_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Set page config
st.set_page_config(
    page_title="ImageMind Classifier",
    page_icon=os.path.join(SCRIPT_DIR, "assets/hazz.png"),
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF;
        }
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        .logo-image {
            width: 45px;
            height: 45px;
            object-fit: contain;
        }
        .title-text {
            color: #E86C52;
            font-size: 2.5em;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .description {
            text-align: center;
            margin: 30px 0;
            color: #2C3E50;
        }
        div[data-testid="stFileUploader"] {
            width: 100%;
        }
        button[data-testid="baseButton-secondary"] {
            background-color: #FFFFFF;
            border-color: #E86C52;
            color: #E86C52;
            border-radius: 5px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(232, 108, 82, 0.1);
            margin-top: 20px;
        }
        .confidence-meter {
            width: 100%;
            height: 10px;
            background-color: #E5E5E5;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background-color: #E86C52;
            border-radius: 5px;
            transition: width 0.5s ease-in-out;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Center column for content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Container for logo and title
    logo_base64 = get_base64_encoded_image("assets/hazz.png")
    st.markdown(f"""
        <div class='title-container'>
            <img src='data:image/png;base64,{logo_base64}' class='logo-image'>
            <h1 class='title-text'>ImageMind</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Subtitle
    st.markdown(
        "<div style='text-align: center; color: #2C3E50; font-size: 1.2em; font-weight: 500; margin-bottom: 20px;'>Image Classifier</div>",
        unsafe_allow_html=True
    )
    
    # Description with proper spacing
    st.markdown(
        "<div class='description'>Upload any image and our AI will classify it into one of 10 categories using deep learning.</div>",
        unsafe_allow_html=True
    )

# File uploader
uploaded_file = st.file_uploader(
    "Drop an image here",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except:
        st.error("⚠️ Model file not found. Please ensure the model is trained and saved as 'model.h5'")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    # Resize image to 32x32 pixels (CIFAR-10 format)
    image = image.resize((32, 32))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array

# Load the model
model = load_model()

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    with col2:
        with st.spinner('Analyzing image...'):
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Get prediction and apply softmax
            prediction = model.predict(processed_image)
            probabilities = tf.nn.softmax(prediction).numpy()[0]
            
            predicted_class = class_names[np.argmax(probabilities)]
            confidence = float(np.max(probabilities)) * 100
            
            # Display results with enhanced UI
            st.markdown("""
                <div class='prediction-box' style='padding: 20px; border-radius: 10px; background-color: rgba(232, 108, 82, 0.1); width: 100%;'>
                    <div style='display: flex; align-items: center;'>
                        <span style='font-size: 1.4em; font-weight: 600; color: #2C3E50;'>
                            🎯 Prediction Results
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Main prediction
            st.markdown(f"""
                <div style='font-size: 1.2em; font-weight: 600; color: #E86C52;'>
                    {predicted_class.title()}
                </div>
                <div class='confidence-meter'>
                    <div class='confidence-fill' style='width: {confidence}%;'></div>
                </div>
                <div style='text-align: right; font-size: 0.9em; color: #666;'>
                    Confidence: {confidence:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            # Top 3 predictions header and list
            st.markdown("""
                <div style='margin-top: 20px;'>
                    <div style='font-weight: 600; margin-bottom: 10px;'>Top 3 Predictions</div>
            """, unsafe_allow_html=True)
            
            # Top 3 predictions list
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            for idx in top_3_indices:
                prob = probabilities[idx] * 100
                st.markdown(f"""
                    <div class='prediction-item' style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span style='color: #2C3E50;'>{class_names[idx].title()}</span>
                        <span style='color: #666;'>{prob:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# Add information about the model
with st.expander("ℹ️ About the Model"):
    st.markdown("""
        <div style='white-space: normal; word-wrap: break-word; max-width: 100%;'>
            <p style='margin-bottom: 15px;'>
                This model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes:
            </p>
            <div style='font-family: monospace; padding-left: 20px; margin-bottom: 15px;'>
                ✈️ Airplane<br>
                🚗 Automobile<br>
                🐦 Bird<br>
                🐱 Cat<br>
                🦌 Deer<br>
                🐕 Dog<br>
                🐸 Frog<br>
                🐎 Horse<br>
                🚢 Ship<br>
                🚛 Truck
            </div>
            <p style='white-space: normal; word-wrap: break-word;'>
                The model uses a deep convolutional neural network architecture to achieve high accuracy in classifying these images.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Made with ❤️ by Hazz using Streamlit and TensorFlow
    </div>
""", unsafe_allow_html=True)