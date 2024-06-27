from google.colab import drive
drive.mount('/content/drive')

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16
from keras.models import Model

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import randint
import pandas as pd
import pickle

base_dir = '/content/drive/MyDrive/'
train_dir='/content/drive/MyDrive/'
train_type1_dir='/content/drive/MyDrive/'
train_type5_dir='/content/drive/MyDrive/'
train_type10_dir='/content/drive/MyDrive/'
train_type11_dir='/content/drive/MyDrive/'
train_typecfb2_dir='/content/drive/MyDrive/'
test_dir='/content/drive/MyDrive/'
test_type1_dir='/content/drive/MyDrive/'
test_type5_dir='/content/drive/MyDrive/'
test_type10_dir='/content/drive/MyDrive/'
test_type11_dir='/content/drive/MyDrive/'
test_typecfb2_dir='/content/drive/MyDrive/'

train1=len(os.listdir(train_type1_dir))
train5=len(os.listdir(train_type5_dir))
train10=len(os.listdir(train_type10_dir))
train11=len(os.listdir(train_type11_dir))
traincfb2=len(os.listdir(train_typecfb2_dir))
test1=len(os.listdir(test_type1_dir))
test5=len(os.listdir(test_type5_dir))
test10=len(os.listdir(test_type10_dir))
test11=len(os.listdir(test_type11_dir))
testcfb2=len(os.listdir(test_typecfb2_dir))

total_train=train1+train5+train10+train11+traincfb2
total_test=test1+test5+test10+test11+testcfb2

print(train1,train5,train10,train11,traincfb2,)
print(test1,test5,test10,test11,testcfb2)

path = '/content/drive/MyDrive/'
os.chdir(path)

images = []
with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.png'):
            images.append(file.name)

IMG_SHAPE  = 224
batch_size = 32

image_gen_train = ImageDataGenerator(rescale = 1./255)
train_data_gen = image_gen_train.flow_from_directory(batch_size = batch_size,
directory = train_dir,
shuffle= True,
target_size = (IMG_SHAPE,IMG_SHAPE))

image_gen_test = ImageDataGenerator(rescale=1./255)
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
directory=test_dir,
target_size=(IMG_SHAPE, IMG_SHAPE))

pre_trained_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

pre_trained_model.summary()

for layer in pre_trained_model.layers:
  print(layer.name)
  layer.trainable = False

  last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])

model.summary()

vgg_classifier = model.fit(train_data_gen,
epochs = 30,
verbose = 1)

result = model.evaluate(test_data_gen)
print("test_loss, test accuracy",result)

model_json = model.to_json()
with open("/content/drive/MyDrive/output/Raw_Classifier/VGG_type_OI_Classifier.json", "w") as json_file:
  json_file.write(model_json)
  model.save("/content/drive/MyDrive/output/Raw_Classifier/VGG_type_OI_Classifier.h5")
  print("Saved model to disk")
  model.save_weights("/content/drive/MyDrive/output/Raw_Classifier/VGG_type_OI.h5")

loss_train =vgg_classifier.history['loss']
epochs = range(0,30)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train =vgg_classifier.history['acc']
acc_val = vgg_classifier.history['acc']
epochs = range(0,30)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()