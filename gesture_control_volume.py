import cv2
# import mediapipe as mp 
import time 
import numpy as np
import hand_tracking_module as HTM
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# ----------------------------------------------------------------
cam_W, cam_H = 640, 480
p_time = 0
# ----------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, cam_W)
cap.set(4, cam_H)

detector = HTM.handDetector(detectionConfidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
print("volume range is - ", volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
# (-63.5, 0.0, 0.5)

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 300
volPercentage = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!= 0:
        # print(lmlist[4], lmlist[8])
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (x1, y1),10, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2, y2),10, (255,0,0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy),10, (255,0,0), cv2.FILLED)
        length=math.hypot(x2-x1, y2-y1)
        print(length)

        # hand range 25 , 300
        # volume range -63, 0
        vol = np.interp(length,[50,300],[minVol, maxVol])
        volBar = np.interp(length,[50,300],[300, 150])
        volPercentage = np.interp(length,[50,300],[0, 100])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)
        


        if length < 50:
            # 25 - 300
            cv2.circle(img, (cx, cy),10, (0,255,0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85, 300), (255,0,0), 3)
    cv2.rectangle(img, (50,int(volBar)), (85, 300), (255,0,0), cv2.FILLED)
    cv2.putText(img, f'volume: {int(volPercentage)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 28:20