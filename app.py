# Flask utils
from flask import Flask, redirect, url_for, request, render_template


import numpy as np
import pandas as pd
import os
import random
import datetime
import matplotlib.pyplot as plt
import tifffile as tiff

from keras import backend as K
# from sklearn.metrics import jaccard_similarity_score

from collections import defaultdict
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import gc
import warnings
import zipfile
warnings.filterwarnings("ignore")
from keras.models import load_model
import tensorflow as tf
import random as rn
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tqdm import tqdm


from skimage.transform import resize
from tifffile import imread, imwrite



num_cls = 10
size = 160
smooth = 1e-12


def adjust_contrast(bands, lower_percent=2, higher_percent=98):
    """
    to adjust the contrast of the image 
    bands is the image 
    """
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)





def SegNet():
    size=160
    
    #tf.random.set_seed(32)
    classes= 10
    img_input = Input(shape=(size, size, 8))
    x = img_input
    

    # Encoder 
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 23))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_initializer = tf.keras.initializers.he_normal(seed= 43))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 32))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 41))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 33))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 35))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 54))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 39))(x)
    x = BatchNormalization()(x)
    
    
    #Decoder
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 45))(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 41))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 49))(x)
    x = BatchNormalization()(x)
      
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 18))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 21))(x)
    x = BatchNormalization()(x)
    x = Conv2D(classes, kernel_size=3, activation='relu', padding='same', kernel_initializer = tf.keras.initializers.he_normal(seed= 16))(x)
  
    x = Activation("softmax")(x)
    
    model = Model(img_input, x)
  
    return model





def resize_image(image, model):
  """
  to resize the image
  """
  if image.shape == (837,837,8):
    return image

  else:
    resized_data = resize(image, (837,837,8))
    imwrite('resized.tif', resized_data, planarconfig='CONTIG')
    return tiff.imread("resized.tif")   



import tensorflow as tf
graph = tf.get_default_graph()


def model_predict(img_path, model):
     
    rgb_img = tiff.imread(img_path)
    img = np.rollaxis(rgb_img, 0, 3)

    #resize the image according to model architecture
    img = resize_image(img, model)

    #adjust the contrast of the image
    x = adjust_contrast(img)
    
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((num_cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x
     
    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * size:(i + 1) * size, j * size:(j + 1) * size])
            
        x = 2 * np.transpose(line, (0, 1, 2, 3)) - 1
        with graph.as_default():
            tmp = model.predict(x, batch_size=4)
        tmp = np.transpose(tmp,(0,3,1,2))
        
        for j in range(tmp.shape[0]):
            prd[:, i * size:(i + 1) * size, j * size:(j + 1) * size] = tmp[j]
     
    # thresholds for each class 
    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(num_cls):
        prd[i] = prd[i] > trs[i]
    p = prd[:, :img.shape[0], :img.shape[1]]
    return p    



print('functions')
app=Flask(__name__)

UPLOAD_FOLDER='/home/lohit/Downloads/dstl-satellite-imagery-feature-detection/sixteen_band'
print('functions_2')



@app.route('/', methods=['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        uploaded_image=request.files['image']
        if uploaded_image:
            file_path= os.path.join(UPLOAD_FOLDER,uploaded_image.filename)
            uploaded_image.save(file_path)
            
            mask = model_predict(file_path, model)
            class_list = ["Buildings", "Manmade structures" ,"Road",\
                        "Track","Trees","Crops","Waterway","Standing water",\
                        "Vehicle Large","Vehicle Small"]

            #rgb_img = os.path.join( 'sixteen_band', '{}_M.tif'.format(id)) 
            rgb_img = file_path     
            rgb_img = tiff.imread(rgb_img)
            image = np.rollaxis(rgb_img, 0, 3)
            print(image.shape)
                    
            img = np.zeros((image.shape[0],image.shape[1],3))
            img[:,:,0] = image[:,:,4] #red
            img[:,:,1] = image[:,:,2] #green
            img[:,:,2] = image[:,:,1] #blue

            for i in range(num_cls):
                plt.figure(figsize=(10,10))
                ax1 = plt.subplot(131)
                ax1.set_title('image ID:6120_2_0')
                ax1.imshow(adjust_contrast(img))
                ax2 = plt.subplot(132)
                ax2.set_title("predict "+ class_list[i] +" pixels")
                ax2.imshow(mask[i], cmap=plt.get_cmap('gray'))
                plt.savefig("static/output_"+str(i)+".png")

            return render_template('index.html',prediction_values="Prediction Done ", flag=1)
    return render_template('index.html',prediction_values=0,flag=0)


if __name__=="__main__":

     model = SegNet()
     model.load_weights("/home/lohit/Desktop/DSTL/upload_images/weights-99-0.4161.hdf5")
     app.run(host='0.0.0.0', port=8095)
