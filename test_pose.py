import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV Camera Setup
cap = cv2.VideoCapture(1)  # Change to 0 if using the internal camera

def calculate_angle(a, b, c):
    """Calculate the angle between three points: a (shoulder), b (elbow), c (wrist)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Ensure valid range
    
    return angle

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame for hand tracking and face tracking
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    feedback = "Position your hand correctly"
    h, w, _ = frame.shape
    
    face_landmarks = None
    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            face_landmarks = face.landmark
    
    if hand_results.multi_hand_landmarks and face_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand points
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            # Approximate elbow position (relative to wrist)
            elbow_x = wrist.x - 0.15  
            elbow_y = wrist.y + 0.2  
            elbow_pos = (int(elbow_x * w), int(elbow_y * h))

            # Convert to pixel coordinates
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            index_pos = (int(index_knuckle.x * w), int(index_knuckle.y * h))
            middle_pos = (int(middle_knuckle.x * w), int(middle_knuckle.y * h))

            # Extract face points (forehead, temples, nose)
            forehead = face_landmarks[10]  # Approximate forehead landmark
            temple_left = face_landmarks[93]  # Left temple
            temple_right = face_landmarks[323]  # Right temple
            nose = face_landmarks[1]  # Nose bridge

            forehead_pos = (int(forehead.x * w), int(forehead.y * h))
            temple_left_pos = (int(temple_left.x * w), int(temple_left.y * h))
            temple_right_pos = (int(temple_right.x * w), int(temple_right.y * h))
            nose_pos = (int(nose.x * w), int(nose.y * h))

            # Check Block Validity

            # 1. **Alignment** - Knuckles should align with upper face (forehead/temples)
            if temple_left_pos[1] - 20 <= index_pos[1] <= temple_right_pos[1] + 20:
                alignment_feedback = "Alignment: ✅"
            else:
                alignment_feedback = "Fix hand alignment!"

            # 2. **Distance from Face** - Wrists should be ~5-15 cm from the face
            face_wrist_distance = abs(nose_pos[0] - wrist_pos[0]) + abs(nose_pos[1] - wrist_pos[1])
            if 50 <= face_wrist_distance <= 150:  # Approximate pixel range
                distance_feedback = "Distance: ✅"
            else:
                distance_feedback = "Move hands closer!"

            # 3. **Elbow Position** - Angle should be 80°-110°
            elbow_angle = calculate_angle(elbow_pos, wrist_pos, index_pos)
            if 80 <= elbow_angle <= 110:
                elbow_feedback = "Elbow Position: ✅"
            else:
                elbow_feedback = "Adjust elbow position!"

            # 4. **Hand Rotation** - Back of hands should face outward
            hand_rotation_feedback = "Rotation: ✅"  # Placeholder (needs advanced detection)

            # Generate final feedback
            feedback = f"{alignment_feedback} | {distance_feedback} | {elbow_feedback} | {hand_rotation_feedback}"

            # Draw Stick Figure
            cv2.circle(frame, elbow_pos, 8, (255, 0, 0), -1)  # Elbow (blue)
            cv2.circle(frame, wrist_pos, 8, (0, 255, 0), -1)  # Wrist (green)
            cv2.line(frame, elbow_pos, wrist_pos, (255, 255, 255), 2)  # Arm Line
            cv2.circle(frame, index_pos, 6, (0, 0, 255), -1)  # Index Knuckle (red)
            cv2.circle(frame, middle_pos, 6, (0, 0, 255), -1)  # Middle Knuckle (red)

    # Display feedback
    cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Hand Block Validation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
