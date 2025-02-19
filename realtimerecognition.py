import face_recognition
import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle

# Load the known face encodings, names, and average depths
with open("face_encodings.pkl", "rb") as f:
    known_faces, known_names, average_depths = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Reduce webcam resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize dlib's facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_head_pose(landmarks):
    # Extract key points (eyes, nose)
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])

    # Calculate vectors
    eye_vector = right_eye - left_eye
    nose_vector = nose_tip - ((left_eye + right_eye) / 2)

    # Calculate angles (yaw and pitch)
    yaw = np.arctan2(nose_vector[1], nose_vector[0]) * 180 / np.pi
    pitch = np.arctan2(nose_vector[0], eye_vector[0]) * 180 / np.pi
    return yaw, pitch

def check_perspective_distortion(landmarks):
    # Measure distances between key points
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
    mouth_center = np.array([landmarks.part(66).x, landmarks.part(66).y])

    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_mouth_distance = np.linalg.norm(nose_tip - mouth_center)

    # Compare ratios
    ratio = nose_to_mouth_distance / eye_distance
    return ratio

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Reduced resolution for faster processing
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)

    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5
    B = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_average_depth(depth_map, face_location):
    top, right, bottom, left = face_location

    # Ensure the face region is within the bounds of the depth map
    height, width = depth_map.shape
    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height - 1))
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1))

    # Crop the depth map to the face region
    face_depth = depth_map[top:bottom, left:right]

    # Check if the face region is valid (non-empty)
    if face_depth.size == 0:
        return None

    # Calculate the average depth
    avg_depth = np.mean(face_depth)
    return avg_depth

EYE_AR_THRESH = 0.2  # Threshold for detecting a blink
blink_detected = False
prev_yaw, prev_pitch = None, None
prev_landmarks = None
frame_count = 0  # For frame skipping
confidence_score = 0  # Confidence score for liveness

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Skip frames to reduce processing load
    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame (increased consistency)
        continue

    # Reset confidence score for each frame
    confidence_score = 0

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face with known faces using a stricter threshold
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = face_distances.argmin()
        name = "Unknown"
        if face_distances[best_match_index] < 0.5:  # Stricter threshold
            name = "Face Detected"

        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Perform liveness checks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Check head pose
            yaw, pitch = calculate_head_pose(landmarks)
            if prev_yaw is not None and prev_pitch is not None:
                yaw_diff = abs(yaw - prev_yaw)
                pitch_diff = abs(pitch - prev_pitch)
                if yaw_diff > 5 or pitch_diff > 5:  # Threshold for movement
                    confidence_score += 1
            prev_yaw, prev_pitch = yaw, pitch

            # Check perspective distortion
            ratio = check_perspective_distortion(landmarks)
            if 0.3 < ratio < 0.7:  # Adjusted range for 3D structure
                confidence_score += 1

            # Check depth using MiDaS
            depth_map = estimate_depth(frame)
            detected_depth = calculate_average_depth(depth_map, face_location)
            if detected_depth is not None:
                depth_difference = abs(detected_depth - average_depths[best_match_index])
                if depth_difference < 0.2:  # Adjusted threshold for depth mismatch
                    confidence_score += 1

            # Check for eye blink
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_AR_THRESH:
                blink_detected = True

            if blink_detected:
                confidence_score += 2  # Higher weight for blink detection
                blink_detected = False  # Reset for next detection

            # Check motion parallax
            if prev_landmarks is not None:
                displacement = np.linalg.norm(
                    np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]) -
                    np.array([(prev_landmarks.part(i).x, prev_landmarks.part(i).y) for i in range(68)])
                )
                if displacement > 10:  # Threshold for motion
                    confidence_score += 1
            prev_landmarks = landmarks

        # Confirm liveness based on confidence score
        if confidence_score >= 4:  # Require at least 4 successful checks
            print("Face Successfully Detected")
        else:
            pass  # Suppress all other outputs

    cv2.imshow("Face Unlock", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()