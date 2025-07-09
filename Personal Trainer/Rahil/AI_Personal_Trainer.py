import cv2
import numpy as np
import time
import pose_detection_module as pm

cap = cv2.VideoCapture('C:/Users/nazar/Desktop/VS Code/Projects/Computer_Vison/Pose Estimation/1.mp4')  # Or 0 for webcam

# Optional: resize window dimensions
wCam, hCam = 1280, 720

detector = pm.poseDetector()
count = 0
dir = 0  # 0: down, 1: up
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ← Reset to frame 0
        continue

    img = cv2.resize(img, (wCam, hCam))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Right arm curl: shoulder(12), elbow(14), wrist(16)
        angle = detector.findAngle(img, 12, 14, 16)

        # Convert angle to percentage
        per = np.interp(angle, (50, 150), (0, 100))

        bar = np.interp(per, (0, 100), (400, 50))

        # Check curl direction
        if per >= 99:
            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 10:
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw curl progress bar
        cv2.rectangle(img, (550, 50), (600, 400), (0, 0, 0), 3)
        if bar >100:
            cv2.rectangle(img, (550, int(bar)), (600, 400), (255, 255, 255), cv2.FILLED)
        else:
            cv2.rectangle(img, (550, int(bar)), (600, 400), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (535, 45),   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw count box
        cv2.rectangle(img, (0, 300), (200, 720), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)),(50,450), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 5)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

