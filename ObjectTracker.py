import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"


with open(classFile) as fp:
    labels = fp.read().split("\n")
print(labels)

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

display = True

###############################################################################
#detects image from pretrained model downloaded from tensorflow and
#adds them to a list called objects
#

def detect_objects(net, im):
    dim = 300
    #create a blob from the image
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=False)

    #pass blob to the network
    net.setInput(blob)

    #Perform prediction
    objects = net.forward()
    return objects

################################################################################
#display_text creates text within a black rectangle at the position specifed


def display_text(im, text, x, y):

    #get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    #use text size to create a black rectangle
    cv2.rectangle(frame, (x,y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED);
    #display text inside the rectangle
    cv2.putText(frame, text, (x, y-5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


################################################################################
#display_objects function iterates through detections and gets the position
#of the objects. It then tests the detections against the confidence threshold
#before testing if it is one of the desired objects then finally either creating
#a rectangle around the object or prints out the detected object depending on the
#state of display    

def display_objects(frame, objects, target_labels, threshold = 0.20):
    rows = frame.shape[0]; cols = frame.shape[1]

    #For every Detected object
    for i in range(objects.shape[2]):
        #find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        #recover original coordinates from mormalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        #check if detection is good quality
        if score > threshold:
            for j in range(len(target_labels)):
                if labels[classId] == target_labels[j]:
                    if display == True:
                        display_text(frame, "{}".format(labels[classId]), x, y)
                        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        print(target_labels[j])

#################################################################################
#Main executed code which will detect and display the desired objects
#and either display over the image or just print out what has been detected                        
   
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Video Player"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


while cv2.waitKey(1) != 27: #code for the escape key
    has_frame, frame = source.read()
    if not has_frame:
            break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
       
    objects = detect_objects(net, frame)
    display_objects(frame, objects, ['cell phone', 'dog', 'book'])

    #inference time for detections to be made average around 20ms on intel i9 processor
    t, _ = net.getPerfProfile()
    inference_timer = 'Inference time: %.2f ms' %(t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, inference_timer, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),0)

    if display == True:
        cv2.imshow(win_name, frame)

    key = cv2.waitKey(1)
    if key == ord('f'):
        display = False
    elif key == ord('d'):
        display = True


source.release()
cv2.destroyWindow(win_name)

