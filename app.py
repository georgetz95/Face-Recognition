from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle
from PIL import Image
import os
from utils import pipeline_model
app = Flask(__name__)


upload_folder = 'static/uploads'

def get_width(path):
    image = np.array(Image.open(path))
    size = image.shape
    aspect = size[1] / size[0]
    width = int(300 * aspect)
    return width

@app.route('/base')
def base():
    return render_template('base.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faceapp', methods=['GET', 'POST'])
def faceapp():
    if request.method == 'POST':
        image = request.files['image']
        file_name = image.filename
        save_path = os.path.join(upload_folder, file_name)
        image.save(save_path)
        print(f"Image saved successfully to '{upload_folder}'")
        width = get_width(save_path)
        # Image Prediction
        pipeline_model(save_path, file_name)
        return render_template('faceapp.html', file_upload=True, img_name=file_name, width=width, height=300)
        
    return render_template('faceapp.html', file_upload=False)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)



