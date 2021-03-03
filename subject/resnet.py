# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 09:10:02 2021

@author: 89748
"""

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

TrianImage="../zzz/train/"
ValidImage="../zzz/valid/"
TestImage="../zzz/test/"
NonInfectedimages = os.listdir(TrianImage + "/Non-infected")
Capimages = os.listdir(TrianImage + "/Cap")
Covid19images = os.listdir(TrianImage + "/Covid-19")

#print the size of the dataset
print(len(NonInfectedimages), len(Capimages), len(Covid19images))
NUM_TRAINING_IMAGES = len(NonInfectedimages) + len(Capimages) + len(Covid19images)
print(NUM_TRAINING_IMAGES)

#image size and batch size
image_size = 512
BATCH_SIZE = 16
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

#root path
data_path = '../zzz'

#Dataset Generation
#Using Keras ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True)
valid_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)


#flow from directory method
training_set = train_datagen.flow_from_directory(data_path + '/train',
                                                 color_mode='grayscale',
                                                 target_size = (image_size, image_size),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

valid_set = valid_datagen.flow_from_directory(data_path + '/valid',
                                            color_mode='grayscale',
                                            target_size = (image_size, image_size),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical',
                                            shuffle = True)

testing_set = test_datagen.flow_from_directory(data_path + '/test',
                                            color_mode='grayscale',
                                            target_size = (image_size, image_size),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical',
                                            shuffle = True)

print("train batch ", training_set.__getitem__(0)[0].shape)
print("test batch ", testing_set.__getitem__(0)[0].shape)
print("valid batch ", valid_set.__getitem__(0)[0].shape)
print("sample train label \n", training_set.__getitem__(0)[1][:5])

print(training_set.class_indices)
print(training_set.class_indices)
print(valid_set.class_indices)

labels = ['Cap', 'Covid-19', 'Non-infected']

def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    
from tensorflow import keras

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index.numpy()

def superimposed_img(image, heatmap):
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_size, image_size))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

#applying category smoothing 
def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss

#callbacks
#lr_reduce:learning rate reduction mechanism
#es_callback:Early Stopping
#tb_callBack:tensorboard visualization
#sb_callBack:Save Best Weight(when validation loss minimize)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=8, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, verbose=1)
tb_callBack = TensorBoard(
    log_dir='./model',
    histogram_freq=1, batch_size=32,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500
)
sb_callBack=keras.callbacks.ModelCheckpoint('./save_model/res50best.h5', 
                    monitor='val_loss', 
                    verbose=0, 
                    save_best_only=False, 
                    save_weights_only=False, 
                    mode='auto', 
                    period=1)

#Counter
counter = Counter(training_set.classes)                          
max_val = float(max(counter.values()))

#balance the dataset
#each category have different images
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)

#import the model ResNet50
#train with Adam optimizer(initial lr=0.0006)
#compiled with loss function 'category smoothing',Adam optimizer and accurancy metrics
model = tf.keras.applications.ResNet50(weights= None, include_top=True, input_shape= (512,512,1),classes=3)
optimizer = tf.keras.optimizers.Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
model.compile(loss=categorical_smooth_loss, optimizer=optimizer, metrics=['accuracy'])
model.summary()

#training
model.fit_generator(training_set, validation_data=valid_set, callbacks=[lr_reduce, es_callback, tb_callBack,sb_callBack], epochs=200)  

#model save
model.save("model_res50_ls_0003_09596.h5")

#test on testing set        
Pred = model.evaluate_generator(testing_set, verbose=1)
print(Pred)
