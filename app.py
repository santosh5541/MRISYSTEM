import os
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
# Load model
MODEL_PATH = os.path.join('models', 'mrimodel.h5')
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit uploads to 5MB

def preprocess_image(img: Image.Image, target_size=(128, 128)) -> np.ndarray:
    img = img.resize(target_size)
    img = img.convert('RGB')
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('index.html', error="Please choose an image file.")

        # save upload
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # predict
        img = Image.open(filepath)
        x = preprocess_image(img)
        probs = model.predict(x)[0]

        # build and sort predictions
        predictions = [
            {'label': name, 'confidence': f"{p * 100:.2f}%"}
            for name, p in zip(CLASS_NAMES, probs)
        ]
        predictions.sort(key=lambda x: float(x['confidence'].strip('%')), reverse=True)
        top = predictions[0]

        return render_template(
            'result.html',
            filename=filename,
            top_label=top['label'],
            top_confidence=top['confidence'],
            predictions=predictions
        )

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # for local dev; remove debug=True in production
    app.run(host='0.0.0.0', port=5000, debug=True)
