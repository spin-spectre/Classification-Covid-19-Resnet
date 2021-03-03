import os
import numpy as np
import cv2                 
from random import shuffle
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from collections import Counter
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import load_model

def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


model = load_model("model_res50_ls_0006_09596.h5",custom_objects={'categorical_smooth_loss': categorical_smooth_loss})

res=[]
for i in range(159):
    patient='./testData/T'+str(i).rjust(3,'0')+'/'
    l=os.listdir(patient)
    cap=0
    covid=0
    normal=0
    for f in l:
        if f[0]=='.':
            continue
        img_test=cv2.imread(patient+f,0)
        img_test=img_test/255.0
        img_test=np.expand_dims(img_test, axis=0)
        img_test=np.expand_dims(img_test, axis=3)
        pred=np.argmax(model.predict(img_test),axis=1)
        if pred[0]==0:
            cap=cap+1
        elif pred[0]==1:
            covid=covid+1
        elif pred[0]==2:
            normal=normal+1
    
    tot=cap+covid+normal
    print(cap,covid,normal)
    if max(cap,covid,normal)==cap:
        res.append('Cap')
    elif max(cap,covid,normal)==covid:
        res.append('Covid-19')
    elif max(cap,covid,normal)==normal:
        res.append('Non-infected')
        
print(res)
    
    



