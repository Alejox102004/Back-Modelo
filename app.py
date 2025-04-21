
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://wondrous-sprinkles-3ef632.netlify.app"}})

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modelo_digit_classification_mejorado.h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            "predicted_class": predicted_class,
            "probabilities": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
