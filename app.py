import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Maize Leaf Disease Detection", page_icon="🌽")

# 2. Load the Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_maize_model():
    # Make sure the filename matches your .h5 file exactly
    return tf.keras.models.load_model("maize_leaf_disease_model.keras")

model = load_maize_model()

# 3. Class Labels (Must match train_data.class_indices)
classes = {
    0: "Grey Leaf Spot",
    1: "Healthy",
    2: "Leaf Blight",
    3: "Rust"
}

st.title("🌽 Maize Leaf Disease Detector")
st.write("Upload an image of a maize leaf to identify potential diseases.")

# 4. Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Home", "Dataset Info", "Settings"])

if menu == "Home":
    # 5. File Uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
        
        # 6. Preprocessing & Prediction
        if st.button("Analyze Image"):
            with st.spinner("Classifying..."):
                # Preprocess the image to match MobileNetV2 requirements
                img = image.convert('RGB') 
                img = img.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                # MobileNetV2 uses preprocess_input (normalization)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                class_id = np.argmax(predictions)
                confidence = np.max(predictions) * 100

                # 7. Display Results
                st.subheader("Result:")
                if classes[class_id] == "Healthy":
                    st.success(f"{classes[class_id]} ({confidence:.2f}% Confidence)")
                else:
                    st.warning(f"Detected: {classes[class_id]} ({confidence:.2f}% Confidence)")
                
                # Practical advice based on prediction
                if class_id == 3: # Rust
                    st.info("Advice: Apply recommended fungicides and check for high humidity.")
                elif class_id == 2: # Leaf Blight
                    st.info("Advice: Remove infected debris and improve air circulation.")

elif menu == "Dataset Info":
    st.subheader("About the Model")
    st.write("This model was trained using Transfer Learning on **MobileNetV2**.")
    st.write("Target Diseases: Grey Leaf Spot, Leaf Blight, and Rust.")

elif menu == "Settings":
    st.write("App Version 1.0.0")