import numpy as np
import cv2
#192.168.225.52
#192.168.43.185
cap = cv2.VideoCapture("rtsp://192.168.43.185:81/stream")

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
