## dataset https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import os               
import numpy as np  
import tflearn 
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

PATH_ = "Kaggle/Facial_Expression_Recognition/"
def Build_Model(width, height, classes, img_channel, lr):
    convnet = input_data(shape=[None, width, height, img_channel], name='input')

    convnet = conv_2d(convnet, 32, (3, 3), activation='relu',strides=1)
    convnet = max_pool_2d(convnet, (2, 2), strides=3)

    convnet = conv_2d(convnet, 64, (3, 3), activation='relu')
    convnet = max_pool_2d(convnet, (2, 2))
    
    convnet = conv_2d(convnet, 64, (3, 3), activation='relu')
    convnet = max_pool_2d(convnet, (2, 2))

    convnet = conv_2d(convnet, 128, (3, 3), activation='relu')
    convnet = max_pool_2d(convnet, (2, 2))
    
    convnet = conv_2d(convnet, 128, (3, 3), activation='relu')
    convnet = max_pool_2d(convnet, (2, 2))

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5)
   
    convnet = fully_connected(convnet, classes, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir= PATH_+ 'log')
    return model