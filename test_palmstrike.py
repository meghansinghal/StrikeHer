import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Load reference image (correct posture)
reference_image = "palmstrike.jpg"

# Open webcam feed for the user
cap_user = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap_user.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the reference image and process it with MediaPipe
ref_img = cv2.imread(reference_image)
ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_results = pose.process(ref_img_rgb)

# Feedback Queue System
feedback_queue = []
feedback_colors = []

# Function to provide feedback for correct posture
def give_feedback(ref_landmarks, user_landmarks):
    feedback = []
    feedback_colors.clear()

    # Ensure both reference and user landmarks are available
    if ref_landmarks and user_landmarks:
        # Extract relevant landmarks (shoulders, wrists, etc.)

        # Left side
        ref_left_shoulder = ref_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        user_left_shoulder = user_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        ref_left_wrist = ref_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        user_left_wrist = user_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Right side
        ref_right_shoulder = ref_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        user_right_shoulder = user_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        ref_right_wrist = ref_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        user_right_wrist = user_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Check left shoulder position relative to the reference (to ensure correct posture)
        shoulder_diff_left = abs(ref_left_shoulder.x - user_left_shoulder.x)
        if shoulder_diff_left > 0.05:
            feedback.append(f"Left shoulder off by {round(shoulder_diff_left * 100, 2)}%. Move left shoulder {'left' if user_left_shoulder.x < ref_left_shoulder.x else 'right'}.")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append(f"Left shoulder aligned with reference ({round(shoulder_diff_left * 100, 2)}%)")
            feedback_colors.append((0, 255, 0))  # Green for correct

        # Check left wrist position relative to shoulder
        wrist_diff_left = abs(ref_left_wrist.x - user_left_wrist.x)
        if wrist_diff_left > 0.05:
            feedback.append(f"Left wrist off by {round(wrist_diff_left * 100, 2)}%. Move left wrist {'left' if user_left_wrist.x < ref_left_wrist.x else 'right'}.")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append(f"Left wrist aligned with reference ({round(wrist_diff_left * 100, 2)}%)")
            feedback_colors.append((0, 255, 0))  # Green for correct

        # Check right shoulder position relative to the reference (to ensure correct posture)
        shoulder_diff_right = abs(ref_right_shoulder.x - user_right_shoulder.x)
        if shoulder_diff_right > 0.05:
            feedback.append(f"Right shoulder off by {round(shoulder_diff_right * 100, 2)}%. Move right shoulder {'left' if user_right_shoulder.x < ref_right_shoulder.x else 'right'}.")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append(f"Right shoulder aligned with reference ({round(shoulder_diff_right * 100, 2)}%)")
            feedback_colors.append((0, 255, 0))  # Green for correct

        # Check right wrist position relative to shoulder
        wrist_diff_right = abs(ref_right_wrist.x - user_right_wrist.x)
        if wrist_diff_right > 0.05:
            feedback.append(f"Right wrist off by {round(wrist_diff_right * 100, 2)}%. Move right wrist {'left' if user_right_wrist.x < ref_right_wrist.x else 'right'}.")
            feedback_colors.append((0, 0, 255))  # Red for incorrect
        else:
            feedback.append(f"Right wrist aligned with reference ({round(wrist_diff_right * 100, 2)}%)")
            feedback_colors.append((0, 255, 0))  # Green for correct

    return feedback

# FPS Calculation
prev_time = 0

# Timer for 2 minutes (120 seconds)
start_time = time.time()

# Loop through the webcam feed
while cap_user.isOpened():
    ret_user, frame_user = cap_user.read()
    
    if not ret_user:
        print("Error: Failed to capture frame.")
        break

    # Flip frame horizontally for a mirror effect
    frame_user = cv2.flip(frame_user, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    user_results = pose.process(rgb_user)

    # Draw landmarks on user frame
    if user_results.pose_landmarks:
        mp_draw.draw_landmarks(frame_user, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    # Provide real-time feedback for the user
    feedback_texts = give_feedback(ref_results.pose_landmarks, user_results.pose_landmarks)

    # Display feedback on the user frame
    for i, text in enumerate(feedback_texts):
        cv2.putText(frame_user, text, (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_colors[i], 2)

    # Show the video
    cv2.imshow("User Posture Feedback", frame_user)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame_user, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Check if 2 minutes have passed
    if time.time() - start_time > 120:
        cv2.putText(frame_user, "Time's Up! Practice Finished.", (20, 50 + len(feedback_texts) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("2 minutes are up! Practice finished.")
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close windows
cap_user.release()
cv2.destroyAllWindows()
