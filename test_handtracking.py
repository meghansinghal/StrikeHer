import cv2
import mediapipe as mp
import numpy as np
import time

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# Initialize MediaPipe Holistic with segmentation
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=True, smooth_segmentation=True)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# OpenCV Camera Setup
cap = cv2.VideoCapture(1)  # Use internal camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set Full HD resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering for real-time processing

# FPS Calculation
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(rgb_frame)

    if results.segmentation_mask is not None:
        # Convert segmentation mask to 3 channels
        mask = results.segmentation_mask
        mask_3d = np.dstack((mask, mask, mask))

        # Create a blurred background
        blurred_bg = cv2.GaussianBlur(frame, (35, 35), 50)

        # Merge the original frame and blurred background based on the mask
        frame = (frame * mask_3d + blurred_bg * (1 - mask_3d)).astype(np.uint8)

    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show output
    cv2.imshow("Full HD User with Blurred Background", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
