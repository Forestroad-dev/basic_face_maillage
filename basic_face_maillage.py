import cv2
import numpy as np
import mediapipe as mp
import time


cap =cv2.VideoCapture(0)
mpmaille=mp.solutions.face_mesh
vismaille=mpmaille.FaceMesh(max_num_faces=2)
mpdraw = mp.solutions.drawing_utils
drawspec=mpdraw.DrawingSpec((0,255,0),thickness=1,circle_radius=1)
ptime=0
ctime=0
while True:
    succes,img =cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resultats = vismaille.process(imgRGB)
    if resultats.multi_face_landmarks:
        for visage in  resultats.multi_face_landmarks:
             mpdraw.draw_landmarks(img,visage,mpmaille.FACEMESH_CONTOURS,landmark_drawing_spec=drawspec)



    ctime=time.time()
    fps =1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,'FPS='+str(np.int32(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow('Video face',img)
    cv2.waitKey(1)