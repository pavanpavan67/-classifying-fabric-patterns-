from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')
model = load_model(model_path)

labels = ['floral', 'plain', 'paisley', 'striped', 'geometric', 'polka-dotted']  # Update as needed

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Ensure upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            img = load_img(file_path, target_size=(255, 255))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            label = labels[np.argmax(pred)]
            prediction = f"Predicted: {label}"

            # Convert file path to URL-safe path
            image_url = '/' + file_path.replace('\\', '/')

    return render_template('index.html', prediction=prediction, image_url=image_url)


    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            img = load_img(file_path, target_size=(255, 255))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            label = labels[np.argmax(pred)]
            prediction = f"Predicted: {label}"
            image_url = file_path

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)