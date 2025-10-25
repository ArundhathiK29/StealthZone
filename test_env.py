import cv2
import mediapipe as mp
import numpy as np

print("✅ OpenCV version:", cv2.__version__)
print("✅ MediaPipe version:", mp.__version__)

# Try to open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access camera.")
else:
    print("🎥 Camera detected successfully!")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cap.release()

print("✅ All basic checks completed.")