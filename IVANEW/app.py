from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('waste_classification_model.h5')

# Define labels
labels = {0: 'Biodegradable', 1: 'Non-biodegradable'}

# Set up an upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected!", 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image
    image = load_img(file_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(image)
    predicted_class = labels[int(np.round(prediction[0][0]))]

    # Render the result template with prediction
    return render_template('result.html', prediction=predicted_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
