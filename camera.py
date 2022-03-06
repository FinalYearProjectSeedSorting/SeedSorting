# camera.py

import cv2
import PIL.Image
from PIL import Image
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        #self.video = cv2.VideoCapture(0)
        self.video = cv2.VideoCapture('http://192.168.43.185:81/stream')
        self.k=1
        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()
        cv2.imwrite("seed.png", image)
        mm2 = PIL.Image.open("seed.png")
        if self.k>=10:
            rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
            rz.save("seed1.png")
            self.k=1
        
        self.k+=1

            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
