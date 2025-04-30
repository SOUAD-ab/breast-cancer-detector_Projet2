import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définition des chemins pour les répertoires d'images
train_dir = "Dataset_BUSI_with_GT/train"
test_dir = "Dataset_BUSI_with_GT/test"

# Normalisation et augmentation des images
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalisation
    rotation_range=20,       # Rotation aléatoire
    zoom_range=0.2,          # Zoom aléatoire
    horizontal_flip=True,    # Flip horizontal
    validation_split=0.2     # 20% des données pour validation
)

# Chargement des données d'entraînement
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # 2 classes : cancer / normal
    subset='training'
)

# Chargement des données de validation
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Définition du modèle CNN
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),  # Utilisation de InputLayer pour la première couche
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 sortie : cancer ou normal
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(train_data, validation_data=val_data, epochs=10)

# Sauvegarde du modèle
model.save("model/cancer_detector.h5")

print("Modèle entraîné et sauvegardé ✅")
