import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Titre
st.set_page_config(page_title="Détection de Cancer", layout="centered")  # adapte à l'écran
st.title("🔬 Détection du Cancer du Sein")

# Charger le modèle
model = load_model('model/cancer_detector.h5')

# Uploader une image
uploaded_image = st.file_uploader("📤 Téléchargez une image échographique", type=["jpg", "png"])

# Bouton de prédiction
if uploaded_image is not None:
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    st.image(img, caption="🖼️ Image téléchargée", use_container_width=True)

    if st.button("🔎 Prédire"):
        prediction = model.predict(img_array)
        label = "⚠️ Malignant (cancer probable)" if prediction[0] > 0.5 else "✅ Benign (pas de cancer)"
        st.subheader("Résultat de la Prédiction :")
        st.success(label if "Benign" in label else label, icon="✅" if "Benign" in label else "⚠️")
