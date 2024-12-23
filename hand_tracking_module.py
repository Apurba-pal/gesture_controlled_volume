import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False , maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5 ):

        # static_image_mode = False,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5,
        # max_num_hands = 2

        self.mode = mode
        self.maxHands = int(maxHands)
        self.detectionConfidence = float(detectionConfidence)
        self.trackConfidence = float(trackConfidence)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLmark in self.results.multi_hand_landmarks:
                if draw:                
                    self.mpDraw.draw_landmarks(img, handLmark, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                # print(id, landmark)
                h, w, c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                # print(id,cx,cy)
                landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy),5,(255,0,0), cv2.FILLED)
        return landmarkList



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList)!=0:
            print(landmarkList[4])

            # basic code to manipulate fps
        cTime =time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()














if __name__ == '__main__':
    main()