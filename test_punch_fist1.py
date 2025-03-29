import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Holistic model (for hand tracking and background segmentation)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=True, smooth_segmentation=True)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Load reference video (correct punch from one angle)
reference_video = "punch_fist_correct11.mp4"

# Open reference video and user's camera feed
cap_ref = cv2.VideoCapture(reference_video)
cap_user = cv2.VideoCapture(1)

# Get reference video properties (assuming the video is well-formed)
ref_fps = int(cap_ref.get(cv2.CAP_PROP_FPS))
ref_width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
ref_height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize webcam feed to match reference video size
cap_user.set(3, ref_width)
cap_user.set(4, ref_height)

# Feedback Queue System
feedback_queue = []
feedback_colors = []

# Function to provide feedback for correct posture
def give_feedback(ref_landmarks, user_landmarks):
    feedback = []
    feedback_colors.clear()
    
    # Ensure both reference and user landmarks are available
    if ref_landmarks and user_landmarks:
        # Extract relevant landmarks (left hand, wrist, shoulder, elbow)
        ref_left_shoulder = ref_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        user_left_shoulder = user_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        ref_left_wrist = ref_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        user_left_wrist = user_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Check wrist position relative to shoulder (to ensure punch posture)
        if abs(ref_left_shoulder.x - user_left_shoulder.x) > 0.05:
            feedback.append("Move your left shoulder into position")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append("Left shoulder in position")
            feedback_colors.append((0, 255, 0))  # Green for correct

        if abs(ref_left_wrist.x - user_left_wrist.x) > 0.05:
            feedback.append("Align your left wrist with shoulder")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append("Left wrist aligned")
            feedback_colors.append((0, 255, 0))  # Green for correct

    return feedback

# FPS Calculation
prev_time = 0

# Loop through the reference video and webcam feed
while cap_ref.isOpened() and cap_user.isOpened():
    ret_ref, frame_ref = cap_ref.read()
    ret_user, frame_user = cap_user.read()

    if not ret_ref or not ret_user:
        break

    # Flip frame horizontally for a mirror effect
    frame_user = cv2.flip(frame_user, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
    rgb_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose and Holistic
    ref_results = pose.process(rgb_ref)
    user_results = holistic.process(rgb_user)

    # Draw landmarks on reference video (pose landmarks)
    if ref_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_ref, ref_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    # Draw hand landmarks on user frame
    if user_results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))
    if user_results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))

    # Draw pose landmarks on user frame
    if user_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    # Provide real-time feedback for the user
    feedback_texts = give_feedback(ref_results.pose_landmarks, user_results.pose_landmarks)

    # Display feedback on the user frame
    for i, text in enumerate(feedback_texts):
        cv2.putText(frame_user, text, (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_colors[i], 2)

    # Check if posture is correct (no red feedback)
    if not any(color == (0, 0, 255) for color in feedback_colors):
        cv2.putText(frame_user, "Posture Correct", (20, 50 + len(feedback_texts) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("Posture is correct!")
        break  # Exit loop if posture is correct

    # Combine both frames side-by-side
    frame_user = cv2.resize(frame_user, (frame_ref.shape[1], frame_ref.shape[0]))  # Resize user frame
    combined_frame = np.hstack((frame_ref, frame_user))

    # Show the video
    cv2.imshow("Reference vs User", combined_frame)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame_user, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close windows
cap_ref.release()
cap_user.release()
cv2.destroyAllWindows()
