from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('vgg16-improved-weights.h5')  # Load the model
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Define class labels

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # HTML form to upload image

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Ensure the directory exists
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        predicted_class = model_predict(file_path, model)
        return render_template('result.html', result=predicted_class)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]  # Get the index of the max class score
    return labels[class_idx]  # Return the label corresponding to the predicted index

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)  # Optionally change the port
