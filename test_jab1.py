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
cap_user = cv2.VideoCapture(1)

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

    # Draw landmarks on both videos with minimal overlay
    if ref_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_ref, ref_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3),  # Red for reference
                               mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))  # Yellow for reference

    if user_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),  # Green for user
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))  # Cyan for user

    # Provide real-time feedback
    feedback_texts = give_detailed_feedback(ref_results.pose_landmarks, user_results.pose_landmarks)

    # Display feedback with color-coded text based on severity
    y_offset = 50
    for feedback in feedback_texts:
        # Color coding for feedback severity
        if "good" in feedback:
            color = (0, 255, 0)  # Green for good posture
        elif "minor" in feedback:
            color = (0, 255, 255)  # Yellow for minor adjustments
        elif "major" in feedback:
            color = (0, 0, 255)  # Red for major posture issues
        else:
            color = (255, 0, 0)  # Blue for technique tips

        # Display feedback text with the color
        cv2.putText(frame_user, feedback, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        y_offset += 40

    # Combine both frames side-by-side
    frame_user = cv2.resize(frame_user, (frame_ref.shape[1], frame_ref.shape[0]))  # Resize user frame
    combined_frame = np.hstack((frame_ref, frame_user))

    # Resize the window to a standard size (1024x768 for better visibility)
    cv2.namedWindow("Reference vs User", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reference vs User", 1024, 768)  # Resize window to 1024x768

    # Show the video
    cv2.imshow("Reference vs User", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_ref.release()
cap_user.release()
cv2.destroyAllWindows()





