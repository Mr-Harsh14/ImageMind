import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64

# Function to load and encode the image
def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    return None

# Check if assets directory exists, if not create it
if not os.path.exists('assets'):
    os.makedirs('assets')

# Set page config
st.set_page_config(
    page_title="Image Classifier",
    page_icon="assets/hazz.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container */
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2rem 0 3rem 0;
        animation: fadeIn 0.8s ease-in;
    }
    
    .hazz-img {
        width: 60px;
        height: 60px;
        margin-right: 15px;
    }
    
    /* Main content */
    .main-content {
        padding: 0 20px;
    }
    
    /* Upload box */
    .upload-box {
        border: 2px dashed #E86C52;
        border-radius: 16px;
        background: linear-gradient(145deg, #FFF8F6 0%, #FFFFFF 100%);
        box-shadow: 0 4px 6px rgba(232, 108, 82, 0.1);
        transition: all 0.3s ease;
        margin: 2rem 0;
        padding: 0;
    }
    
    .upload-box:hover {
        box-shadow: 0 6px 8px rgba(232, 108, 82, 0.15);
        border-color: #D55941;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(145deg, #FFF8F6 0%, #FFFFFF 100%);
        padding: 30px;
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid rgba(232, 108, 82, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        animation: slideUp 0.5s ease-out;
    }
    
    /* Typography */
    h1 {
        color: #E86C52 !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #2C3E50;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Buttons and interactive elements */
    .stButton>button {
        background: linear-gradient(135deg, #E86C52 0%, #D55941 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(232, 108, 82, 0.2);
        background: linear-gradient(135deg, #D55941 0%, #C54830 100%);
    }
    
    /* File uploader customization */
    .stUploadDropzone {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    /* Remove default borders and styling */
    [data-testid="stFileUploader"] > section {
        border: none !important;
    }
    
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
        border: none !important;
    }

    .css-1v0mbdj {
        border: none !important;
    }

    /* Style the actual upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed #E86C52 !important;
        border-radius: 16px !important;
        background: #FFF8F6 !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }

    /* Upload dropzone */
    [data-testid="stFileUploadDropzone"] {
        min-height: 150px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: transparent !important;
        border: none !important;
        padding: 1rem !important;
    }

    /* Style the browse files button */
    button[data-testid="stUploadDropzoneButton"] {
        background: #E86C52 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 4px 12px !important;
    }

    /* Hide default text */
    [data-testid="stFileUploader"] label {
        display: none !important;
    }

    /* Style the file type text */
    [data-testid="stFileUploader"] small {
        color: #666 !important;
    }

    /* Remove any extra margins/padding */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Remove any extra padding/margins */
    .element-container, .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
    }

    .upload-box {
        border: 2px dashed #E86C52 !important;
        border-radius: 16px !important;
        background: #FFF8F6 !important;
        padding: 0 !important;
        margin: 1rem 0 !important;
        overflow: hidden !important;
    }

    /* Hide default upload text */
    .css-1erivf3 {
        display: none !important;
    }

    /* File type info text */
    .css-d1b1ld {
        margin-top: 0.5rem !important;
        color: #666 !important;
    }

    /* Expander customization */
    .streamlit-expanderHeader {
        background-color: #FFF8F6;
        border-radius: 8px;
        border: 1px solid rgba(232, 108, 82, 0.2);
        padding: 10px 15px;
        font-weight: 500;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #E86C52;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Confidence meter styling */
    .confidence-meter {
        background: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #E86C52 0%, #D55941 100%);
        transition: width 0.5s ease-out;
    }
    
    /* Top predictions list */
    .prediction-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(232, 108, 82, 0.1);
    }
    
    .prediction-item:last-child {
        border-bottom: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        
        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with logo
logo_path = 'assets/hazz.png'
encoded_logo = encode_image(logo_path)

if encoded_logo is not None:
    st.markdown(f"""
        <div class="title-container">
            <img src="data:image/png;base64,{encoded_logo}" class="hazz-img">
            <h1>Image Classifier</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("🖼️ Image Classifier")

st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Upload any image and our AI will classify it into one of 10 categories using deep learning.
    </div>
""", unsafe_allow_html=True)

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