import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load your saved model
model = load_model('Model/Chess_model_v1.h5')  # Update the path to your saved model

def main():
    st.title('Chess Position (FEN) Recognition')
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    with col2:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img = img.resize((25, 25)) 
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])

            class_labels = {
                0: 'FEN_Position_0',
                1: 'FEN_Position_1',
                2: 'FEN_Position_2',
                3: 'FEN_Position_3',
                4: 'FEN_Position_4',
                5: 'FEN_Position_5', 
                6: 'FEN_Position_6',
                7: 'FEN_Position_7',
                8: 'FEN_Position_8', 
                9: 'FEN_Position_9',
                10: 'FEN_Position_10',
                11: 'FEN_Position_11',
                12: 'FEN_Position_12'
            }
            st.subheader("Prediction")
            st.write(f"Model Predicted FEN Position: {class_labels[predicted_class]}")

if __name__ == '__main__':
    main()
