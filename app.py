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

# Set page config with custom logo
if os.path.exists('assets/hazz.png'):
    st.set_page_config(
        page_title="Image Classifier",
        page_icon="assets/hazz.png",
        layout="centered"
    )
else:
    st.set_page_config(
        page_title="Image Classifier",
        page_icon="🔍",
        layout="centered"
    )

# Custom CSS for better styling - updated with logo colors
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #E86C52;  /* Orange color from logo */
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: #FFF8F6;  /* Light orange background */
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #E86C52;
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .hazz-img {
        width: 50px;
        height: 50px;
        margin-right: 10px;
    }
    h1 {
        color: #E86C52 !important;
    }
    .stButton>button {
        background-color: #E86C52;
        color: white;
    }
    .stButton>button:hover {
        background-color: #D55941;
        color: white;
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
    This app uses a deep convolutional neural network trained on the CIFAR-10 dataset 
    to classify images into 10 different categories.
""")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except:
        st.error("⚠️ Model file not found. Please ensure the model is trained and saved as 'model.h5'")
        return None

def preprocess_image(image):
    # Resize image to 32x32 pixels
    image = image.resize((32, 32))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array

# Load the model
model = load_model()

# File uploader
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    if model is not None:
        with col2:
            with st.spinner('Analyzing image...'):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Get prediction and apply softmax
                prediction = model.predict(processed_image)
                probabilities = tf.nn.softmax(prediction).numpy()[0]
                
                predicted_class = class_names[np.argmax(probabilities)]
                confidence = float(np.max(probabilities)) * 100
                
                # Display results
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("### Prediction Results")
                st.markdown(f"**Class:** {predicted_class.title()}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Display top 3 predictions
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                st.markdown("#### Top 3 Predictions:")
                for idx in top_3_indices:
                    st.markdown(f"- {class_names[idx].title()}: {probabilities[idx]*100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

# Add information about the model
with st.expander("ℹ️ About the Model"):
    st.markdown("""
        This model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images 
        in 10 different classes. The classes are:
        
        - Airplane
        - Automobile
        - Bird
        - Cat
        - Deer
        - Dog
        - Frog
        - Horse
        - Ship
        - Truck
        
        The model uses a deep convolutional neural network architecture to achieve high accuracy 
        in classifying these images.
    """)

# Footer
st.markdown("---")
st.markdown("Made by Hazz using Streamlit and TensorFlow")