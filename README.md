# DermaNet Diagnostic System: Skin Condition Detection using Machine Learning
Welcome to DermaNet, a machine learning-based web application designed to assist in detecting skin conditions from dermatology images. This project uses a Convolutional Neural Network (CNN) to classify skin conditions, focusing on detecting common lesions such as melanoma.

This project was built using TensorFlow, Keras, OpenCV, and deployed via a simple Flask backend. The web interface is developed using React, and the entire project is hosted via GitHub Pages.

## Project Overview
DermaNet is a complete full-stack project that integrates:

* React frontend to upload images for analysis
* Flask backend to process and predict skin conditions
* Machine Learning Model: A CNN-based model using MobileNetV2 trained on the publicly available HAM10000 dermatology dataset*
* Image Preprocessing: OpenCV was used to preprocess images before passing them to the model for predictions

## Features

Upload an image of a skin lesion, and the model will predict whether it is positive or negative.The user interface is intuitive, clean, and responsive, designed with modern UI principles. Backend handles real-time predictions using a pre-trained deep learning model. The model was trained using data augmentation techniques to ensure robust generalization.

## Tech Stack

* Frontend: React, CSS for styling
* Backend: Flask, TensorFlow, Keras
* ML Model: MobileNetV2 (pre-trained on ImageNet, fine-tuned on the HAM10000 dataset)
* Image Processing: OpenCV for resizing and normalizing images
* Deployment: GitHub Pages for the frontend, Flask backend on a local server

# How It Works

* Image Upload: Users can upload an image of a skin lesion via the web interface.
* Image Processing: The image is preprocessed (resized and normalized) before being passed to the machine learning model.
* Prediction: The model processes the image and returns a classification result (e.g., benign or malignant).
* Result Display: The result is displayed on the web page along with the confidence score of the prediction.

# Dataset

The model was trained using the HAM10000 dataset, which is a collection of high-quality dermatoscopic images of skin lesions. 

# Future Improvements

MAJOR CHANGE INCOMING: Code Documentation and Comments

* Additional Skin Conditions: Expand the model to classify more types of skin lesions.
* Mobile Compatibility: Make the web app more optimized for mobile users.
* Model Fine-Tuning: Further improve accuracy with hyperparameter tuning and model architecture adjustments.

