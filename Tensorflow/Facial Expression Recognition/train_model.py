####
__author__ = 'AM7'

## dataset https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
####

import os
import cv2                 
import numpy as np                     
from random import shuffle 
from tqdm import tqdm      

import random
from sklearn import preprocessing
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import imutils 
from sklearn.model_selection import train_test_split
from keras.applications import imagenet_utils
import math
import pickle
form model import Build_Model

PATH_ = "Kaggle/Facial_Expression_Recognition/"
LR = 1e-3 # 0.001
EPOCHS = 12
IMG_SIZE = 48

MODEL_NAME = PATH_ + '_Facial-{}-{}.model'.format(LR, 'CNN')
TRAIN_DIR = PATH_ + 'Training'
TEST_DIR = PATH_ + 'PrivateTest'
data = []
labels = []
##  (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).


imagePaths = sorted(list(paths.list_images(TRAIN_DIR)))
random.seed(9001)
random.shuffle(imagePaths)

def process_image(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

def create_train_data(imagePaths):
    for imagPath in tqdm(imagePaths):
        img = process_image(imagPath)
        data.append(img)

        label =  imagPath.split(os.path.sep)[-2]
        labels.append(label)
    
    np.save(PATH_ + '_data.npy', data)
    np.save(PATH_ + '_labels.npy', labels)

if os.path.exists(PATH_ + '_data.npy') and os.path.exists(PATH_ + '_labels.npy'):
    data = np.load(PATH_ + '_data.npy')
    labels = np.load(PATH_ + '_labels.npy')
else:
    create_train_data(imagePaths)

data = np.array(data)
labels = np.array(labels,dtype='int')

LB = LabelBinarizer()
labels = LB.fit_transform(labels)
print(LB.classes_)

f = open(PATH_ + 'LB', "wb")
f.write(pickle.dumps(LB))
f.close()

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

trainX = np.array(trainX).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
testX = np.array(testX).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

model = Build_Model(IMG_SIZE, IMG_SIZE, len(LB.classes_), 1, LR)
print(len(trainX))
if os.path.exists(PATH_ + '_Facial-0.001-CNN.model.meta'):
    model.load(MODEL_NAME)
    mlb = pickle.loads(open(PATH_ + 'LB', "rb").read())
    print('model loaded!')
else:
    print('Building model...')
    model.fit({'input': trainX}, {'targets': trainY}, n_epoch=EPOCHS, validation_set=({'input': testX}, {'targets': testY}), 
        snapshot_step=len(trainX) // 32, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)
eva = model.evaluate(testX, testY,len(trainX) // 32)
print(eva)
