import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🖼️ Deep CNN Image Classifier")
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
        model = tf.keras.models.load_model('notebook/model.h5')
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
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    if model is not None:
        with col2:
            with st.spinner('Analyzing image...'):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Get prediction
                prediction = model.predict(processed_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction)) * 100
                
                # Display results
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("### Prediction Results")
                st.markdown(f"**Class:** {predicted_class.title()}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Display top 3 predictions
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                st.markdown("#### Top 3 Predictions:")
                for idx in top_3_indices:
                    st.markdown(f"- {class_names[idx].title()}: {prediction[0][idx]*100:.2f}%")
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
st.markdown("Made with ❤️ using Streamlit and TensorFlow") 