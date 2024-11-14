from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import opening, closing, square
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULTS_FOLDER'] = 'static/results/'
app.config['MODEL_PATH'] = 'model/classifier-resnet-weights.hdf5'
app.config['SEG_MODEL_PATH'] = 'model/workingResUNet-weights.hdf5'
app.secret_key = 'super secret key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf 

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

custom_objects = {
    'focal_tversky': focal_tversky,
    'dsc': dsc,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'tp': tp,
    'tn': tn,
    'tversky': tversky,
    'tversky_loss': tversky_loss
}

classification_model = load_model(app.config['MODEL_PATH'])
segmentation_model = load_model(app.config['SEG_MODEL_PATH'], custom_objects=custom_objects)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert TIFF to JPEG for web compatibility if needed
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            img = io.imread(file_path)
            filename = filename.rsplit('.', 1)[0] + '.jpg'
            file_path_jpg = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            io.imsave(file_path_jpg, img)
            file_path = file_path_jpg

        tumor_pred, mask_path = process_image(file_path)
        results_filename = os.path.basename(mask_path)
        return render_template('result1.html', original_img=filename, tumor_pred=tumor_pred, results_img=results_filename)
    return render_template('index1.html')

def process_image(file_path):
    img = load_img(file_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    tumor_pred = classification_model.predict(img_array)[0]
    prediction = segmentation_model.predict(img_array)
    predicted_mask = (prediction.squeeze() > 0.0001).astype(int)
    cleaned_mask = closing(opening(predicted_mask, square(3)), square(5))

    mask_path = os.path.join(app.config['RESULTS_FOLDER'], os.path.basename(file_path))
    plt.imsave(mask_path, cleaned_mask, cmap='gray')
    return tumor_pred, mask_path

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def send_result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False)
