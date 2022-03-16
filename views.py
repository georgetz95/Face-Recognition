from flask import render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import os
from utils import pipeline_model


upload_folder = 'static/uploads'

def get_width(path):
    image = np.array(Image.open(path))
    size = image.shape
    aspect = size[1] / size[0]
    width = int(300 * aspect)
    return width


def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

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

def about():
    return render_template('about.html')



