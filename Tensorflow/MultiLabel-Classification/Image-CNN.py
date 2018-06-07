import os
import cv2                 
import numpy as np                     
from random import shuffle 
import matplotlib.pyplot as plt
from tqdm import tqdm      
import tflearn 
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from imutils import paths
import imutils 
from sklearn.model_selection import train_test_split
from keras.applications import imagenet_utils
import math

PATH_ = "dataset"
LR = 1e-3 # 0.001
EPOCHS = 2
IMG_SIZE = 96

MODEL_NAME = '_clothsCategories-{}-{}.model'.format(LR, 'CNN')

imagePaths = sorted(list(paths.list_images(PATH_)))
random.seed(9001)
random.shuffle(imagePaths)

data = []
labels = []
dictt = {
    ("black", "jeans"):  [1, 0, 0 ,   1, 0, 0],
    ("blue", "dress"):  [0, 1, 0 ,   0, 1, 0],
    ("blue", "jeans"):  [0, 1, 0 ,   1, 0, 0],
    ("blue", "shirt"):  [0, 1, 0 ,   0, 0, 1],
    ("red", "dress"):  [0, 0, 1 ,   0, 1, 0],
    ("red", "shirt"):  [0, 0, 1 ,   0, 0, 1]
}

def label_img(imgName):
    (color, clothsCateg) =  imgName.split(os.path.sep)[-2].split("_")
    for k,v in dictt.items():
        if k == (color, clothsCateg):
            return v

def process_image(imagePath):
    img = cv2.imread(imagePath, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype='float').reshape(-1, IMG_SIZE,IMG_SIZE, 3)

    return img

def create_train_data(imagePaths):
    for imagPath in tqdm(imagePaths):
        img = process_image(imagPath)
        data.append(img)

        label = label_img(imagPath)
        labels.append(label)
    
    np.save('_data.npy', data)
    np.save('_labels.npy', labels)

def Build_Model(width, height, classes, img_channel, lr):
    convnet = input_data(shape=[None, width, height, img_channel], name='input')

    convnet = conv_2d(convnet, 32, (3, 3), activation='relu',strides=1)
    convnet = max_pool_2d(convnet, (3, 3), strides=3)

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
   
    convnet = fully_connected(convnet, classes, activation='sigmoid')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model


print("loading images...")
if os.path.exists('_data.npy') and os.path.exists('_labels.npy'):
    data = np.load('_data.npy')
    labels = np.load('_labels.npy')
else:
    create_train_data(imagePaths)

data = np.array(data, dtype='float') / 250.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

trainX = np.array(trainX).reshape(-1, IMG_SIZE,IMG_SIZE, 3)
testX = np.array(testX).reshape(-1, IMG_SIZE,IMG_SIZE, 3)

model = Build_Model(IMG_SIZE, IMG_SIZE, 6, 3, LR)
if os.path.exists('_clothsCategories-0.001-CNN.model.meta'):
    model.load(MODEL_NAME)
    print('model loaded!')
else:
    print('Building model...')
    model.fit({'input': trainX}, {'targets': trainY}, n_epoch=EPOCHS, validation_set=({'input': testX}, {'targets': testY}), 
        snapshot_step=50, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

TEST = 'test/'
fig = plt.figure()

## return the index of 1
def index(idxs):
    if idxs[0] == 0 and idxs[1] == 3:
        return ("black", "jeans")
    elif idxs[0] == 1 and idxs[1] == 4:
        return("blue", "dress")
    elif idxs[0] == 1 and idxs[1] == 3:
        return("blue", "jeans")
    elif idxs[0] == 1 and idxs[1] == 5:
        return("blue", "shirt")
    elif idxs[0] == 2 and idxs[1] == 4:
        return("red", "dress")
    elif idxs[0] == 2 and idxs[1] == 5:
        return("red", "shirt")

for i, imgName in enumerate(os.listdir(TEST)):
    img = cv2.imread(TEST + imgName, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgtmp = img
    img = np.array(img, dtype='float').reshape(-1, IMG_SIZE,IMG_SIZE, 3)
    img = np.array(img, dtype='float') / 250.0

    y = fig.add_subplot(1,3,i+1) 
    ##  predictetion
    pred = model.predict(img)[0]
    ## get the index of 2 max number
    idxs = sorted(np.argsort(pred)[::-1][:2])
    indx = index(idxs)
    
    label = []
    for i in range(len(indx)):
        label.append("{}: {:.2f}%".format( indx[i], pred[idxs[i]] * 100))

    y.imshow(cv2.cvtColor(imgtmp, cv2.COLOR_BGR2RGB))
    plt.title(label[0]+'\n'+label[1])
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

