import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Get TFLite interpreter input and output details
def get_tflite_input_output_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

# Preprocess image function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    if img.shape[-1] == 4:  # Convert RGBA to RGB if needed
        img = img[..., :3]
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict using TFLite model
def predict_with_tflite(interpreter, input_image):
    input_details, output_details = get_tflite_input_output_details(interpreter)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Define class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Streamlit app interface
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to identify potential plant diseases and get recommendations.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    st.write("🔍 **Analyzing the image...**")
    
    # Load the TFLite model
    interpreter = load_tflite_model('model.tflite')
    
    # Preprocess the image
    input_image = preprocess_image(image)
    
    # Predict disease
    predicted_class, confidence = predict_with_tflite(interpreter, input_image)
    predicted_disease = class_labels[predicted_class]
    
    # Display the result
    st.success(f"**Predicted Disease:** {predicted_disease}")
    st.info(f"**Confidence Level:** {confidence:.2f}%")
    
    # Additional information or advice
    st.write("🔗 **What to do next?**")
    st.markdown(
        f"""
        - If the plant is healthy, continue regular care practices.
        - If a disease is detected, take action:
          - Use appropriate fungicides or pesticides.
          - Consult an agricultural expert for specific recommendations.
        """
    )
