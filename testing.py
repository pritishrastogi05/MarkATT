#Importing necessary packages
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
print("All packages imported")

# Loading the model in

model = torch.hub.load("ultralytics/yolov5",'yolov5s')
print("Model loaded successfully")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO',np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release()
cv2.detroyAllWindows()

