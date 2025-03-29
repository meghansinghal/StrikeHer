import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Input and Output Video Paths
input_video_path = "fist.mp4"  # Change this to your input video file
output_video_path = "fist_output.mp4"  # Processed video output file

# Open the video file
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Set up video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # End of video

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(rgb_frame)

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

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Processed Video", frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
