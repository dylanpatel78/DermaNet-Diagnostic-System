from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('dermatology_model_final.keras')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"name": "DermIdentify", "version": "1.0"})

@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Preprocess the image using OpenCV
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction using the loaded model
    prediction = model.predict(img)

    # Assuming the model outputs a softmax array, where index 0 is "Negative" and index 1 is "Positive"
    predicted_class = np.argmax(prediction[0])  # Get the index of the max value
    confidence = prediction[0][predicted_class]  # Get the confidence score

    # Based on class index, define label
    label = "Positive" if predicted_class == 1 else "Negative"

    # Return both prediction and confidence score
    return jsonify({"result": label, "confidence": float(confidence)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
