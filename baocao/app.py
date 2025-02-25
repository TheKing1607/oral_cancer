import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import StandardScaler
import cv2
import os
import tensorflow as tf 
import shutil
from sklearn.cluster import KMeans

st.title("Cancer Diagnosis”")
st.write("Please, upload an image file.png, ...")

file_uploader = st.file_uploader("Please, select a image: ..... ")

if file_uploader:
    image = Image.open(file_uploader)
    st.image(image)


def minImage(foler_path):

    file_list = os.listdir(foler_path)

    image_files = [file for file in file_list if file.endswith(('.jpg', '.png', "webp"))]

    smallest_size = float('inf')
    smallest_image_path = None

    for image_file in image_files:
        image_path = os.path.join(foler_path, image_file)
        image = cv2.imread(image_path)
        
        height, width, _ = image.shape
        
        if height * width < smallest_size:
            smallest_size = height * width
            smallest_image_path = image_path

    return smallest_image_path

class_names = ['cancer', 'non-cancer']

if st.button('Predict'):
    
    if file_uploader is None:
        st.write("Image not detected")
    else:
        image_path = file_uploader
        os.mkdir("Temp")
        image = load_img(image_path, target_size = (50, 50))
        image = img_to_array(image)
        x, y, z = image.shape
        image_2d = image.reshape(x*y, z)
        # data = image_lab.reshape((-1, 3))
        scaler = StandardScaler()
        data = scaler.fit_transform(image_2d)
        kmeans = KMeans(n_clusters=12)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        for i, center in enumerate(centers):
            mask = labels.reshape(image.shape[:2]) == i
            mask = mask.astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(mask)
            cropImg = image[y: y + h, x : x + w]
            cv2.imwrite("Temp/test"+ str(i) +".png", cropImg)
        path_min_image = minImage("Temp")
        print(path_min_image)
        image = load_img(path_min_image, target_size = (50, 50))
        image = load_img(image_path, target_size = (50, 50))
        image_array = img_to_array(image)
        scale_img = np.expand_dims(image_array, axis=0)

        model2 = load_model("model.h5")
        pred = model2.predict(scale_img)
        print(pred)
        output = class_names[np.argmax(pred)]

        import time
        my_bar = st.progress(0)
        with st.spinner("Predicting"):
            time.sleep(1)
        
        st.title(f"Status: {output} with rate {100 * np.round(pred[0][np.argmax(pred)], 5)} %" )
        shutil.rmtree("Temp")
