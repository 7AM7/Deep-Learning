from tqdm import tqdm     
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from model import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import cv2
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

PATH_ = dir_path + '\dataset'
print(PATH_)
LR = 1e-3 # 0.001
EPOCHS = 10
IMG_SIZE = 96
BS = 32

MODEL_NAME = '_mobile-{}-{}.model'.format(LR, 'CNN')

imagePaths = sorted(list(paths.list_images(PATH_)))
random.seed(9001)
random.shuffle(imagePaths)

data = []
labels = []

def label_img(img):
	word_label =  img.split(os.path.sep)[-2]
	if word_label == 'IPhone': return [1,0]
	elif word_label == 'Samsung': return [0,1]

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

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")
	
model = Model.build(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=2, finalAct="softmax")



if os.path.exists('_mobile-0.001-CNN.model.meta'):
    model.load_model(MODEL_NAME)
    print('model loaded!')
else:
	print('Building model...')
	# initialize the optimizer
	opt = Adam(lr=LR, decay=LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
	
	model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
	
	model.save(MODEL_NAME)
