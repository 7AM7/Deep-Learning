import matplotlib
from tqdm import tqdm     
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from model import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

fig = plt.figure()
dir_path = os.path.dirname(os.path.realpath(__file__))
LR = 1e-3 # 0.001 
MODEL_NAME = dir_path + '\_mobile-{}-{}.model'.format(LR, 'CNN')
TEST =  dir_path + r'\test'
IMG_SIZE = 96
model = load_model(MODEL_NAME)
for i, imgName in enumerate(os.listdir(TEST)):
	print(TEST + "\\"+imgName)
	img = cv2.imread(TEST + "\\"+ imgName, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	imgtmp = img
	img = np.array(img, dtype='float').reshape(-1, IMG_SIZE,IMG_SIZE, 3)
	img = np.array(img, dtype='float') / 250.0

	y = fig.add_subplot(4,4,i+1) 
	##  predictetion
	pred = model.predict(img)[0]
	mx = np.argmax(pred)
	if mx == 1: str_label = 'Samsung'
	else: str_label = 'IPhone'
    
	y.imshow(cv2.cvtColor(imgtmp, cv2.COLOR_BGR2RGB))
	label = "{}: {:.2f}%".format(str_label, pred[mx] * 100)
	plt.title(label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()
