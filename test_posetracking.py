import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# OpenCV Camera Setup
cap = cv2.VideoCapture(1)  # Change to 0 if using the internal camera

# FPS Calculation
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(rgb_frame)

    # Create a copy to apply blur for background effect
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 30)

    # Overlay the blurred background onto the original frame
    frame = cv2.addWeighted(frame, 0.6, blurred_frame, 0.4, 0)

    # Draw hand landmarks with custom colors
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    # Draw pose landmarks (full body) with thicker lines
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on screen
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the OpenCV window
    cv2.imshow("Enhanced Hand & Pose Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()

# is bgr to rob converted
# is the frame resized to improve speed? 
# improve quality of video too 