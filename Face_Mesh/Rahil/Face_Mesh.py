import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)
ptime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0),thickness = 1, circle_radius = 1)

while True :
    success ,img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                h,w,c =img.shape
                x,y = int(lm.x*w),int(lm.y*h)
                print(id,x,y)

    ctime =time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)