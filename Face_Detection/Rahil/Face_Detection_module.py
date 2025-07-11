import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self , minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection   #fACE DETECT WITHOUT GOING TO ITS DETAILS
        self.mpDraw = mp.solutions.drawing_utils  #DRAW WITHOUT GOING TO ITS DETAILS
        self.facedetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        

    def findfaces(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        bboxs = []
        self.results =self.facedetection.process(imgRGB)
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                h,w,c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w) , int(bboxC.ymin *h) ,\
                        int(bboxC.width * w) , int(bboxC.height *h) 
                bboxs.append((id, bbox, detection.score))
                # cv2.rectangle(img,bbox,(255,0,255),2)
                if draw:
                    img = self.fancybox(img,bbox)
                    cv2.putText(img,f' {int(detection.score[0]*100)}% ',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            return img, bboxs
    

    def fancybox(self,img,bbox,l=30,t=5,rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top Left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img




def main():
    
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success , img = cap.read()
        img ,bboxs =detector.findfaces(img)
        if len(bboxs) == 0:
            print("No face detected")
        cTime = time.time()
        fps = 1/(cTime -ptime)
        ptime = cTime
        
        cv2.putText(img,f'fps : {(int(fps))} ',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()