import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Détection Cancer du Sein", page_icon="🎗️", layout="centered")

# --- Bandeau avec logos ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    um5_logo = Image.open("images/logo_um5.png")
    st.image(um5_logo, width=200)

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
    pink_logo = Image.open("images/breast.png")
    st.image(pink_logo, width=200)

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

if uploaded_file is not None:
    st.image(uploaded_file, caption='Image importée', use_container_width=True)
    # Ici tu peux ajouter ton modèle et la prédiction
    st.success("✅ Image reçue. Prédiction en cours...")
    # load_model & prédiction à ajouter ici

# --- Lien vers GitHub ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>🔗 <a href='https://github.com/SOUAD-ab/breast-cancer-detector' target='_blank'>Voir le code source sur GitHub</a></p>",
    unsafe_allow_html=True
)
