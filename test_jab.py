# test_fist.py
import cv2
import mediapipe as mp
import numpy as np
import time
from feedback import give_detailed_feedback  # Import the feedback system

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Load reference video (jab_output.mp4)
reference_video = "jab_output.mp4"
cap_ref = cv2.VideoCapture(reference_video)

# Open user's camera feed
cap_user = cv2.VideoCapture(0)

# Get reference video properties
ref_fps = int(cap_ref.get(cv2.CAP_PROP_FPS))
ref_width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
ref_height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize webcam feed to match reference video size
cap_user.set(3, ref_width)
cap_user.set(4, ref_height)

# Loop through both videos
while cap_ref.isOpened() and cap_user.isOpened():
    ret_ref, frame_ref = cap_ref.read()
    ret_user, frame_user = cap_user.read()

    if not ret_ref or not ret_user:
        break

    # Convert to RGB for MediaPipe processing
    rgb_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
    rgb_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    ref_results = pose.process(rgb_ref)
    user_results = pose.process(rgb_user)

    # Draw landmarks on both videos
    if ref_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_ref, ref_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    if user_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    # Provide real-time feedback
    feedback_texts = give_detailed_feedback(ref_results.pose_landmarks, user_results.pose_landmarks)
    
    # Display feedback on user frame
    for i, text in enumerate(feedback_texts):
        cv2.putText(frame_user, text, (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Combine both frames side-by-side
    frame_user = cv2.resize(frame_user, (frame_ref.shape[1], frame_ref.shape[0]))  # Resize user frame
    combined_frame = np.hstack((frame_ref, frame_user))

    # Show the video
    cv2.imshow("Reference vs User", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_ref.release()
cap_user.release()
cv2.destroyAllWindows()
