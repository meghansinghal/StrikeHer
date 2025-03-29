import cv2
import numpy as np

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Couldn't access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display both original and grayscale video
    cv2.imshow("Original", frame)
    # cv2.imshow("Grayscale", gray)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

