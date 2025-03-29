import cv2

cap = cv2.VideoCapture(1)  # Try changing 0 to 1 if you have an external webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
