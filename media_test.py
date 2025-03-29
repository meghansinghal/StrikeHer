import cv2
import mediapipe as mp
import numpy as np
import os


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_joint_angles(landmarks):
    if landmarks:
        joints = {
            "left_elbow": [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            ],
            "right_elbow": [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            ],
            "left_knee": [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ],
            "right_knee": [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ]
        }
        angles = {joint: calculate_angle(*[(p.x, p.y) for p in points]) for joint, points in joints.items()}
        return angles
    return None


def provide_feedback(image_angles, webcam_angles):
    feedback = []
    if image_angles and webcam_angles:
        for joint in image_angles.keys():
            difference = abs(image_angles[joint] - webcam_angles[joint])
            if difference > 15:
                if webcam_angles[joint] > image_angles[joint]:
                    feedback.append(f"Lower your {joint.replace('_', ' ')}.")
                else:
                    feedback.append(f"Raise your {joint.replace('_', ' ')}.")
        if not feedback:
            return "Good job, correct posture!", (0, 255, 0)  # Green color for correct posture
        return " | ".join(feedback), (0, 0, 255)  # Red color for incorrect posture
    return "No posture detected.", (0, 0, 255)  # Red color for no posture detected


# Load list of reference files (both images and videos)
media_folder = "reference_media"  # Folder with reference images and videos
media_files = sorted(os.listdir(media_folder))  # Get all files sorted
current_media_index = 0


# Open webcam
cap_webcam = cv2.VideoCapture(1)


# Open a video capture once for video files (to prevent repeated opening)
cap_video = None


while cap_webcam.isOpened():
    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_webcam:
        break


    # Load the current reference file
    media_path = os.path.join(media_folder, media_files[current_media_index])
    file_extension = os.path.splitext(media_path)[1].lower()


    if file_extension in ['.jpg', '.jpeg', '.png']:  # If it's an image
        image = cv2.imread(media_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_image = pose.process(image_rgb)


        # Extract angles from reference image
        image_angles = extract_joint_angles(results_image.pose_landmarks.landmark) if results_image.pose_landmarks else None


        # Process webcam frame
        frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
        results_webcam = pose.process(frame_webcam_rgb)


        if results_image.pose_landmarks:
            mp_drawing.draw_landmarks(image, results_image.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results_webcam.pose_landmarks:
            mp_drawing.draw_landmarks(frame_webcam, results_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Extract angles from webcam
        webcam_angles = extract_joint_angles(results_webcam.pose_landmarks.landmark) if results_webcam.pose_landmarks else None


        # Provide feedback
        feedback, color = provide_feedback(image_angles, webcam_angles)
        cv2.putText(frame_webcam, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        # Resize image to match webcam frame
        image_resized = cv2.resize(image, (frame_webcam.shape[1], frame_webcam.shape[0]))


        # Combine image and webcam frame
        combined_frame = np.hstack((image_resized, frame_webcam))
        cv2.imshow("Pose Comparison", combined_frame)


    elif file_extension in ['.mp4', '.avi']:  # If it's a video
        if cap_video is None:
            cap_video = cv2.VideoCapture(media_path)  # Open the video file only once


        ret_video, frame_video = cap_video.read()
        if not ret_video:
            cap_video.release()  # Release video capture if end is reached
            cap_video = None  # Reset video capture to open next video file when needed
            current_media_index += 1  # Move to the next media file
            continue


        # Process reference video frame
        video_rgb = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
        results_video = pose.process(video_rgb)


        # Extract angles from reference video
        video_angles = extract_joint_angles(results_video.pose_landmarks.landmark) if results_video.pose_landmarks else None


        # Process webcam frame
        frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
        results_webcam = pose.process(frame_webcam_rgb)


        if results_video.pose_landmarks:
            mp_drawing.draw_landmarks(frame_video, results_video.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results_webcam.pose_landmarks:
            mp_drawing.draw_landmarks(frame_webcam, results_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Extract angles from webcam
        webcam_angles = extract_joint_angles(results_webcam.pose_landmarks.landmark) if results_webcam.pose_landmarks else None


        # Provide feedback
        feedback, color = provide_feedback(video_angles, webcam_angles)
        cv2.putText(frame_webcam, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        # Resize video frame to match webcam frame
        frame_video_resized = cv2.resize(frame_video, (frame_webcam.shape[1], frame_webcam.shape[0]))


        # Combine video frame and webcam frame
        combined_frame = np.hstack((frame_video_resized, frame_webcam))
        cv2.imshow("Pose Comparison", combined_frame)


    # Move to next media file after feedback
    if feedback == "Good job, correct posture!":
        current_media_index += 1  # Proceed to the next file


    if current_media_index >= len(media_files):  # If there are no more files, exit
        cv2.putText(frame_webcam, "All poses completed!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Pose Comparison", frame_webcam)
        cv2.waitKey(3000)  # Show completion message for 3 seconds
        break  # Exit the loop


    # Wait for the 'q' key to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap_webcam.release()
cv2.destroyAllWindows()
