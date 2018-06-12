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
# eva = model.evaluate(testX, testY,len(trainX) // 32)
# print(eva)
image = cv2.imread('Kaggle/Facial_Expression_Recognition//3/me1.jpg')
tmpimg = image
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = imutils.resize(tmpimg, width=400)
 
# pre-process the image for classification
image = cv2.resize(image, (48, 48))
# image = image.astype("float") / 255.0
image = np.array(image).reshape(-1, IMG_SIZE,IMG_SIZE, 1)


proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]
##  (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
for (i, j) in enumerate(idxs):
    strr = ""
    # build the label and draw the label on the image
    if mlb.classes_[j] == 0: strr = "Angry"
    elif mlb.classes_[j] == 1: strr = "Disgust"
    elif mlb.classes_[j] == 2: strr = "Fear"
    elif mlb.classes_[j] == 3: strr = "Happy"
    elif mlb.classes_[j] == 4: strr = "Sad"
    elif mlb.classes_[j] == 5: strr = "Surprise"
    elif mlb.classes_[j] == 6: strr = "Neutral"
        
    label = "{}: {:.2f}%".format(strr, proba[j] * 100)
    cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 ,0, 0), 2)
    
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
 
# show the output image
cv2.imshow("1", output)
cv2.waitKey(0)
