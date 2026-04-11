import cv2 as cv
import time 
import os


os.makedirs("dataset/images", exist_ok=True)
capture = cv.VideoCapture(0)
compteur = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    cv.imshow("Webcam", frame)
    
    key = cv.waitKey(1)
    if key == ord('s'):
        filename = f"dataset/images/image_{compteur:03d}.jpg"
        cv.imwrite(filename, frame)
        print(f"Images sauvegardées dans : {os.path.abspath('dataset/images')}")
        compteur += 1
    elif key == ord('q'):
        break 