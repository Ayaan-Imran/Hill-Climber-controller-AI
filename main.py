import handDetection
import cv2
from pynput.keyboard import Key, Controller

webcam = cv2.VideoCapture(0)
detector = handDetection.HandRecogniser(max_num_hands=1)
keyboard = Controller()

while True:
    success, image = webcam.read()

    processed_image, hand_landmark_pos = detector.findHands(image, draw=False)
    cv2.imshow("Webcam footage with AI", processed_image)

    try:
        if detector.matchFingerState(image, [True, True, True, True, True]):
            keyboard.release(Key.left)
            keyboard.press(Key.right)

        else:
            keyboard.release(Key.right)
            keyboard.press(Key.left)

    except ValueError:
        print("\nERROR!")
        print("1. Show your full hand")
        print("2. Show only 1 hand in TOTAL.")
        print("3. Show Increase your lighting and go to a better environment.")

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
