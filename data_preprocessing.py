import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Définition des chemins
train_dir =  "Dataset_BUSI_with_GT/train"
test_dir =  "Dataset_BUSI_with_GT/test"

# Normalisation et augmentation des images
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalisation
    rotation_range=20,       # Rotation aléatoire
    zoom_range=0.2,          # Zoom aléatoire
    horizontal_flip=True,    # Flip horizontal
    validation_split=0.2     # 20% des données pour validation
)

# Chargement des images
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # 2 classes : cancer / normal
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Prétraitement terminé ✅")
