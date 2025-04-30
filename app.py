import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modèle
model = load_model('model/cancer_detector.h5')

# Interface Streamlit
st.title("Détection de Cancer - Modèle de Classification")

uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "png"])

if uploaded_image is not None:
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prédiction
    prediction = model.predict(img_array)
    st.image(img, caption="Image téléchargée", use_container_width=True)

    if prediction[0] > 0.5:
        st.error("Résultat : Malignant (cancer détecté)")
    else:
        st.success("Résultat : Benign (pas de cancer détecté)")
