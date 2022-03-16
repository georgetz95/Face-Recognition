import numpy as np
import pandas as pd
import cv2
import pickle

# Loading models
haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('./models/model_best.pkl', 'rb'))
model_pca = pickle.load(open('./models/pca_100.pkl', 'rb'))


def pipeline_model(path, file_name, color='bgr', haar=haar, model_svm=model_svm, model_pca=model_pca):
    genders = ['Female', 'Male']
    image = cv2.imread(path)
    
    if len(image.shape) == 3:
        if color == 'rgb':
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif color == 'bgr':
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:
        if color == 'rgb':
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif color == 'bgr':
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
    face = haar.detectMultiScale(image_gray, 1.2, 9)

    for x, y, w, h in face:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) # Drawing rectangle
        image_crop = image_gray[y:y+h, x:x+w]
        image_crop = image_crop / 255.0
        
        if image_crop.shape[1] > 100:
            image_resize = cv2.resize(image_crop, (100,100), cv2.INTER_AREA)
        else:
            image_resize = cv2.resize(image_crop, (100,100), cv2.INTER_CUBIC)
            
        image_reshape = image_resize.reshape(1,-1)
        image_eigen = model_pca.transform(image_reshape)
        
        results = model_svm.predict_proba(image_eigen)[0]
        predict = results.argmax()
        score = results[predict]
        text = f"{genders[predict]}: {np.round(score,2)}"
        
        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    predict_path = f"./static/predict/{file_name}"
    cv2.imwrite(predict_path, image)
    
    