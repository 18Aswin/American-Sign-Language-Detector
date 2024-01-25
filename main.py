import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5","labels.txt")

offset = 25
size = 300
count = 0

folder = "Images/C"
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    success, img= capture.read()   #Reads the image
    output = img.copy()
    hands, img = detector.findHands(img) #Detect hands in the video/image
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        background = np.ones((size,size,3), np.uint8)*255   #Creating a numpy array for the background of the image to always be a square
        cropped = img[y-offset:y+h+offset, x-offset:x+w+offset]    #Cropping the image of hand only with a fixed offset

        cropshape = cropped.shape

        ratio = h/w
        if ratio>1:    #If height is bigger than width, then we set the height as maximum and adjust width accordingly
            k = size/h
            w_calc = math.ceil(k*w)
            img_resize = cv2.resize(cropped,(w_calc,size))
            img_resizeShape = img_resize .shape
            wGap = math.ceil((size - w_calc)/2)
            background[:, wGap:w_calc+wGap] = img_resize
            prediction, index = classifier.getPrediction(background, draw=False)
            print(prediction,index)

        else:         #If width is bigger than height, then we set the width as maximum and adjust height accordingly
            k = size / w
            h_calc = math.ceil(k * h)
            img_resize = cv2.resize(cropped, (h_calc, size))
            img_resizeShape = img_resize.shape
            hGap = math.ceil((size - h_calc) / 2)
            background[:, hGap:h_calc + hGap] = img_resize
            prediction, index = classifier.getPrediction(background, draw=False)
        cv2.putText(output, labels[index], (x,y-25), cv2.FONT_HERSHEY_PLAIN, 5, (0,102,0),5)
        cv2.rectangle(output,(x-offset,y-offset), (x+w+offset,y+h+offset), (0,102,0), 3)

        cv2.imshow("background",background)
        #cv2.waitKey(1)

    cv2.imshow("Image", output)
    cv2.waitKey(1)    #Gives us a wait time or delay time of 1ms