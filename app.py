import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Cache the model loading for performance (Streamlit >=1.18)
@st.cache_resource
def load_mobilenet_model():
    # Make sure the file path matches where your model is saved
    return load_model("mobilenetv2_base_model.h5")

model = load_mobilenet_model()

# Define your class names and explanations — must match model output classes exactly
class_names = ['Healthy', 'Leaf Rust', 'Berry Disease']
explanations = {
    'Healthy': "This leaf or berry appears healthy. No signs of disease detected.",
    'Leaf Rust': "The model detects signs of leaf rust, a common fungal infection in coffee plants.",
    'Berry Disease': "Signs of berry disease were detected. Consider further inspection and treatment."
}

st.title("Coffee Leaf & Berry Disease Classifier ☕🌿")
st.markdown("Upload an image of a coffee leaf or berry to detect potential diseases.")

uploaded_file = st.file_uploader("Upload a coffee leaf or berry image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Preprocess image for MobileNetV2 model
        img_resized = img.resize((150, 150))  # Resize to match model input
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values to [0,1]

        # Debug info about model shapes
        st.write("Model input shape:", model.input_shape)
        st.write("Model output shape:", model.output_shape)

        # Predict disease class probabilities
        prediction = model.predict(img_array)

        st.write("Raw prediction probabilities:", prediction)

        # Check if model output matches class_names length
        if prediction.shape[1] != len(class_names):
            st.error(f"⚠️ Model predicts {prediction.shape[1]} classes, but you have {len(class_names)} class names.")
        else:
            class_idx = np.argmax(prediction)
            class_name = class_names[class_idx]
            confidence = prediction[0][class_idx]

            st.success(f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)")
            st.markdown(f"**Explanation:** {explanations[class_name]}")

