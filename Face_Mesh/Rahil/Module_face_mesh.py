import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.facemesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(imgRGB)
        faces =[]
        if self.results.multi_face_landmarks:
            
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec, self.drawSpec
                    )
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    h,w,c =img.shape
                    x,y = int(lm.x*w),int(lm.y*h)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img ,faces

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img , faces = detector.findFaceMesh(img)
        # if len(faces)!= 0:
            # print(len(faces))
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS : {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
