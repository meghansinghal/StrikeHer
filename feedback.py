# feedback.py
import numpy as np

def calculate_landmark_distance(landmarks_1, landmarks_2):
    """Calculate the Euclidean distance between corresponding landmarks of two poses."""
    distance = 0
    for i in range(len(landmarks_1)):
        x1, y1 = landmarks_1[i].x, landmarks_1[i].y
        x2, y2 = landmarks_2[i].x, landmarks_2[i].y
        distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def give_detailed_feedback(ref_landmarks, user_landmarks):
    """Generate feedback based on the differences in landmarks between reference and user poses."""
    feedback = []
    
    if not ref_landmarks or not user_landmarks:
        feedback.append("Pose landmarks not detected in one or both frames.")
        return feedback
    
    # Extract pose landmarks
    ref_coords = [(lm.x, lm.y) for lm in ref_landmarks.landmark]
    user_coords = [(lm.x, lm.y) for lm in user_landmarks.landmark]
    
    # Calculate the distance between the corresponding landmarks of both poses
    distance = calculate_landmark_distance(ref_landmarks.landmark, user_landmarks.landmark)

    # Define thresholds for feedback based on distance (tune as necessary)
    if distance < 0.1:
        feedback.append("Great! Your pose is almost identical to the reference.")
    elif distance < 0.3:
        feedback.append("Good job! You're close to the reference pose, but there are some slight misalignments.")
    else:
        feedback.append("Try to align your pose more closely with the reference. Adjust your body position.")

    # Specific joint-based feedback (example for certain key landmarks)
    # For example, checking the position of specific body parts:
    # Adjust this part based on key landmarks you'd like to monitor
    if user_coords[11][1] < ref_coords[11][1]:  # Comparing shoulder positions
        feedback.append("Move your right shoulder down for better alignment.")

    if user_coords[12][1] > ref_coords[12][1]:  # Comparing hip positions
        feedback.append("Your right hip is out of place. Move it back to match the reference pose.")

    # Feedback for specific body parts can be added here:
    # Example for elbows:
    if user_coords[13][0] < ref_coords[13][0]:  # Example comparing elbow position
        feedback.append("Your left elbow is not in line with the reference. Adjust it to match the position.")

    # More specific checks can be added for other parts like wrists, knees, etc.

    return feedback
