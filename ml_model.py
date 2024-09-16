import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load the metadata CSV
metadata = pd.read_csv('./HAM10000_metadata.csv')

# Map the classes from metadata
label_mapping = {'bkl': 0, 'mel': 1}  # Adjust labels as needed for more classes
metadata['label'] = metadata['dx'].map(label_mapping)

# Clean the metadata and ensure the 'label' column is formatted correctly
metadata_cleaned = metadata.dropna(subset=['label'])
metadata_cleaned['label'] = metadata_cleaned['label'].astype(str)

# Paths to image directories (adjust to your environment)
image_dir_part1 = '/Users/dpatel78/Desktop/DermIdentify/HAM10000_images_part_1/'
image_dir_part2 = '/Users/dpatel78/Desktop/DermIdentify/HAM10000_images_part_2/'

# Function to get the image path
def get_image_path(image_id):
    if os.path.exists(os.path.join(image_dir_part1, f'{image_id}.jpg')):
        return os.path.join(image_dir_part1, f'{image_id}.jpg')
    else:
        return os.path.join(image_dir_part2, f'{image_id}.jpg')

metadata_cleaned['image_path'] = metadata_cleaned['image_id'].apply(get_image_path)

# Filter valid image paths
metadata_cleaned = metadata_cleaned[metadata_cleaned['image_path'].apply(os.path.exists)]

# Split into training and validation sets
train_data, val_data = train_test_split(metadata_cleaned, test_size=0.2, stratify=metadata_cleaned['label'], random_state=42)

# Image Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    rotation_range=40,
    width_shift_range=0.3,  
    height_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    x_col='image_path',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Build the CNN model using MobileNetV2 with fine-tuning
def create_model():
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)  # Binary classification

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Fine-tune the last few layers of the base model
    for layer in base_model.layers[-10:]:  # Fine-tune last 10 layers
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Callbacks for early stopping, learning rate reduction, and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_dermatology_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001)

# Handle class imbalance
class_weights = {0: 1.0, 1: 2.0}  # Adjust weights based on class distribution

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // 32,
    validation_data=val_generator,
    validation_steps=len(val_data) // 32,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    class_weight=class_weights  # Apply class weighting
)

# Save the final model
model.save('dermatology_model_final.keras')
from sklearn.metrics import classification_report, confusion_matrix

y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report and confusion matrix
print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))
