# Deep CNN Image Classifier

A modern web application that uses a deep convolutional neural network to classify images into 10 different categories. The model is trained on the CIFAR-10 dataset and deployed using Streamlit for a user-friendly interface.

## Features

- 🖼️ Real-time image classification
- 📊 Confidence scores for predictions
- 🎯 Top 3 predictions display
- 🎨 Modern, responsive UI
- 📱 Mobile-friendly design

## Categories

The model can classify images into the following categories:

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

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Deep-CNN-Image-Classifier.git
cd Deep-CNN-Image-Classifier
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

1. Train the model using the Jupyter notebook:

```bash
jupyter notebook notebook/Image\ Classification.ipynb
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. Open the web application in your browser (default: <http://localhost:8501>)
2. Upload an image using the drag-and-drop interface or file selector
3. Wait for the model to process the image
4. View the predictions and confidence scores

## Project Structure

```
Deep-CNN-Image-Classifier/
├── app.py                  # Streamlit web application
├── notebook/              
│   └── Image Classification.ipynb  # Model training notebook
├── requirements.txt        # Python dependencies
├── model.h5               # Trained model (generated after training)
└── README.md              # Project documentation
```

## Technical Details

- **Framework**: TensorFlow/Keras
- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: CIFAR-10 (60,000 32x32 color images)
- **Frontend**: Streamlit
- **Image Processing**: PIL (Python Imaging Library)

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.24+
- NumPy 1.24+
- Pillow 9.5+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- CIFAR-10 dataset
- TensorFlow team
- Streamlit community
