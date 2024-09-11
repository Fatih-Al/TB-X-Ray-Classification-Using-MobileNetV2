import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


def load_model():
    """
    Load pre-trained deep learning model for carrot freshness classification.
    
    Returns:
        model (tf.keras.Model): Loaded pre-trained deep learning model.
    """
    model = tf.keras.models.load_model('model_TBCMNV2_Smote_5000_10.h5')
    return model

def preprocessing_image(img):
    """
    Preprocesses the input image for model prediction.
    
    Args:
        img: Input image to be preprocessed.
        
    Returns:
        images (numpy.ndarray): Preprocessed image ready for prediction.
    """
    img = img.resize((224, 224))
    images = image.img_to_array(img)
    # images /= 255
    images = np.expand_dims(images, axis=0)
    return images

# # #@st.cache(suppress_st_warning=True)
@st.cache_data
def get_prediction(processed_images):
    """
    Get prediction from the loaded model.
    
    Args:
        processed_images (numpy.ndarray): Preprocessed image for prediction.
        
    Returns:
        classname (str): Predicted class ('Normal' or 'Tuberculosis').
        probability (float): Probability of the predicted class.
    """
    classes = model.predict(processed_images, batch_size=1)
    if classes > 0.5 :
        output_class = 1
    else :
        output_class = 0
    classname = ['Normal', 'Tuberculosis']
    result = classname[output_class]
    output = np.argmax(classes)
    probability = classes[0][output]
    return result, probability

def display_result(result, probability):
    """
    Display prediction result.
    
    Args:
        classname (str): Predicted class ('Normal' or 'Tuberculosis').
        probability (float): Probability of the predicted class.
    """
    st.success('This is a success message!', icon="✅")
    st.image(img, caption='Gambar yang dipilih', use_column_width=False, width=256)
    st.write("Hasil Klasifikasi:")
    if result == "Normal" :
        kelas = ['Normal']
    else :
        kelas = ['Tuberkulosis']
    
    if kelas is ['Normal']:
     st.write(f"Kelas: :green-background{kelas}")
    else:
        st.write(f"Kelas: :red-background{kelas}")  
    
    if probability <= 0.5:
        probability = 100 - (probability / 0.5 * 100)  # Rentang 0 - 0.5
    else:
        probability = ((probability - 0.5) / 0.5 * 100)  # Rentang 0.5 - 1
    

    st.write(f"Probability: {probability:.2f}%")

#Load pre-trained model
model = load_model()

st.title("Aplikasi Klasifikasi :red[Tuberculosis]")

# Upload image through Streamlit
uploaded_file = st.file_uploader("*Masukkan Gambar X-Ray Paru-Paru Anda ...*", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file)
    processed_images = preprocessing_image(img)

    # Predict Image
    result, probability = get_prediction(processed_images)

    # Display prediction result
    display_result(result, probability)

st.write("Develop by :blue-background[Muhammad Syafiq Al Fatih]")
