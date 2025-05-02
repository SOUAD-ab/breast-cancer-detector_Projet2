import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Configuration de la page
st.set_page_config(page_title="Détection Cancer du Sein", page_icon="🎗️", layout="centered")

# --- Bandeau avec logos ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    um5_logo = Image.open("images/logo_um5.png")
    st.image(um5_logo, width=100)

with col2:
    st.markdown(
        "<h1 style='text-align: center; color: #D63384;'>Détecteur de Cancer du Sein</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<h4 style='text-align: center; color: #FF69B4;'>Un projet de Deep Learning pour la santé</h4>", 
        unsafe_allow_html=True
    )

with col3:
    pink_logo = Image.open("breast.png")
    st.image(pink_logo, width=100)

# --- Résumé du projet ---
st.markdown("---")
st.markdown(
    """
    <div style='background-color:#ffe6f0; padding: 15px; border-radius: 10px'>
    <strong>Ce projet</strong> a pour objectif de détecter automatiquement les anomalies sur des images échographiques 
    du sein à l’aide d’un modèle d’intelligence artificielle. Il classe les images en 3 catégories :
    <ul>
        <li><b>Bénin</b> (non dangereux)</li>
        <li><b>Malin</b> (cancer potentiel)</li>
        <li><b>Normal</b> (aucune anomalie)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)

# --- Upload de l'image ---
uploaded_file = st.file_uploader("📤 Veuillez importer une image échographique (format .png ou .jpg)", type=["png", "jpg", "jpeg"])

# Chargement du modèle
@st.cache_resource
def load_trained_model():
    model = load_model("model/model.h5")  # Assurez-vous que le modèle est dans le bon chemin
    return model

model = load_trained_model()

# Si une image est téléchargée, afficher et prédire
if uploaded_file is not None:
    st.image(uploaded_file, caption='Image importée', use_column_width=True)
    
    # Prétraitement de l'image pour la prédiction
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))  # Assurez-vous que la taille correspond à celle du modèle
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisation de l'image

    try:
        # Prédiction du modèle
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        classes = ["Bénin", "Malin", "Normal"]
        result = classes[predicted_class]
        st.markdown(f"### 🩺 Résultat : **{result}**")
        st.success("✅ Prédiction réussie.")
    except Exception as e:
        st.error(f"❌ Erreur de prédiction : {str(e)}")

# --- Lien vers GitHub ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>🔗 <a href='https://github.com/SOUAD-ab/breast-cancer-detector' target='_blank'>Voir le code source sur GitHub</a></p>",
    unsafe_allow_html=True
)
