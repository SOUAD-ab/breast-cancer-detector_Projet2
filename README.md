📌 Projet : Détection du cancer du sein à partir d\'images
échographiques

Ce projet applique un modèle de Deep Learning basé sur le transfer
learning avec MobileNetV2 pour classer des images échographiques du sein
en trois catégories : bénin, malin et normal.

📚 Ce travail a été réalisé dans le cadre du module \"DL Models
Deployment\".

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

📁 1. Dataset utilisé :

🔹 Nom : Dataset_BUSI_with_GT (Kaggle) 🔗 Lien :
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

📸 Nombre total d\'images : 780 🏷️ Classes : • Bénin • Malin • Normal

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

🗂️ 2. Structure du projet :

breast-cancer-detector/ │ ├── 📓 notebook/ │ └──
Breast_Cancer_Classification.ipynb │ ├── 🧠 model/ │ └──
mobilenetv2_model.h5 │ ├── requirements.txt ├── README.md └── 🎥
video_presentation.mp4

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

💻 3. Environnement de développement :

🐍 Python ≥ 3.8

📦 Dépendances principales (requirements.txt) :

\- tensorflow \>= 2.11  - matplotlib  - numpy  - pandas

📥 Installation : \`\`\`bash pip install -r requirements.txt \`\`\`

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

🔄 4. Pipeline du modèle :

🔧 Prétraitement : - Redimensionnement à 224x224 - Normalisation des
pixels (valeurs entre 0 et 1) - Augmentation des données avec
ImageDataGenerator

🧠 Modèle : - MobileNetV2 (pré-entraîné sur ImageNet) - Couches ajoutées
:  - GlobalAveragePooling2D  - Dense(128) + ReLU  - Dropout(0.3)  -
Dense(3) + Softmax

⚙️ Compilation : \`\`\`python optimizer = Adam(learning_rate=1e-4) loss
= \'categorical_crossentropy\' metrics = \[\'accuracy\'\] \`\`\`

🧪 Entraînement : - 20 époques avec validation croisée via val_generator

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

📈 5. Résultats :

✅ Accuracy en validation : \~94 % 📊 Visualisation : courbes d'accuracy
et loss 📉 Évaluation finale : matrice de confusion + classification
report

📷 Image : accuracy_loss_plot.png

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

🎬 6. Vidéo de présentation :

🕒 Durée : 7 minutes Contenu : - Installation des bibliothèques -
Présentation du notebook - Démonstration de l'application

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

📚 7. Ressources utilisées :

\- 📖 Livre Dive into Deep Learning  - 📘 Deep Learning Book (Ian
Goodfellow)  - 🧠 TensorFlow (Keras API)  - 📊 Matplotlib, NumPy,
Pandas, scikit-learn  - 🖼️ OpenCV  - 🌍 Streamlit  - 🗂️ Kaggle (BUSI
Dataset)

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

👥 9. Auteurs :

\- Souad ABOUD  - Abderazzaq NADIR

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

⚖️ 10. Licence :

Projet à usage pédagogique uniquement. 🚫 Non destiné à un usage médical
professionnel.
