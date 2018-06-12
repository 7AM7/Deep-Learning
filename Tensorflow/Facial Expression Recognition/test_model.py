## dataset https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

__author__ = 'AM7'
import os
import cv2                 
import numpy as np  
import imutils 
import dlib
import time
import pickle
from imutils.video import FPS
from imutils import face_utils
from model import Build_Model

PATH_ = "Kaggle/Facial_Expression_Recognition/"
IMG_SIZE = 48
LR = 1e-3 # 0.001
MODEL_NAME = PATH_ + '_Facial-{}-{}.model'.format(LR, 'CNN')

model = Build_Model(IMG_SIZE, IMG_SIZE, 7, 1, LR)
if os.path.exists(PATH_ + '_Facial-0.001-CNN.model.meta'):
    model.load(MODEL_NAME)
    mlb = pickle.loads(open(PATH_ + 'LB', "rb").read())
    print('model loaded!')
else:
     print('model not loaded!')

 

def VideoRecognition():    
    detector = dlib.get_frontal_face_detector()
    video_capture = cv2.VideoCapture(0)
    fps = FPS().start()
    PATH_ = "Kaggle/Facial_Expression_Recognition/"
    while True:
        ret, image = video_capture.read()
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped_image = image[rect.top():rect.bottom() ,rect.left():rect.right()]
            cv2.imshow('cropped', cropped_image)
            cv2.imwrite(PATH_ + str(i) + '.jpg',cropped_image)

            img = cv2.imread('Kaggle/Facial_Expression_Recognition/'+str(i) + '.jpg')
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.array(img).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

            proba = model.predict(img)[0]
            idxs = np.argsort(proba)[::-1][:2]  ## to show first two max proba
            mx = np.argmax(proba)

            strr = checkClasses(mlb.classes_[mx])
            label = "{}: {:.2f}%".format(strr, proba[mx] * 100)
            cv2.putText(image, label, (x - 10, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 ,0, 0), 2)

        cv2.imshow("Output", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
        
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    video_capture.release()

def ImageRecognition():
    image = cv2.imread('Kaggle/Facial_Expression_Recognition/3/me1.jpg')
    tmpimg = image
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = imutils.resize(tmpimg, width=400)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.array(image).reshape(-1, IMG_SIZE,IMG_SIZE, 1)


    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (i, j) in enumerate(idxs):
        strr = checkClasses(mlb.classes_[j])
        label = "{}: {:.2f}%".format(strr, proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 ,0, 0), 2)

    cv2.imshow("1", output)
    cv2.waitKey(0)


def checkClasses(classes):
    if classes == 0: return "Angry"
    elif classes == 1: return "Disgust"
    elif classes == 2: return "Fear"
    elif classes == 3: return "Happy"
    elif classes == 4: return "Sad"
    elif classes == 5: return "Surprise"
    elif classes == 6: return "Neutral"

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


VideoRecognition()
ImageRecognition()