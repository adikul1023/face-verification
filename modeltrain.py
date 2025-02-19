import face_recognition
import cv2
import os
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np

# Load the video
video_path = "face_video.avi"
cap = cv2.VideoCapture(video_path)

# Initialize lists to store face encodings, names, and average depths
known_faces = []
known_names = []
average_depths = []

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)

    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map

def calculate_average_depth(depth_map, face_location):
    top, right, bottom, left = face_location
    face_depth = depth_map[top:bottom, left:right]
    return np.mean(face_depth)

# Extract frames from the video
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 5th frame to reduce redundancy
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Estimate depth map for the frame
    depth_map = estimate_depth(frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        known_faces.append(face_encoding)
        known_names.append("face_0")  # Replace "face_0" with your name or ID

        # Calculate and store the average depth for the face
        avg_depth = calculate_average_depth(depth_map, face_location)
        average_depths.append(avg_depth)

        print(f"Processed frame {frame_count} successfully.")

# Release the video capture
cap.release()

# Save the encodings, names, and average depths to a file
if known_faces and known_names and average_depths:
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_faces, known_names, average_depths), f)
    print("Face encodings and average depths saved successfully!")
else:
    print("No faces were encoded. Please ensure the video contains clear faces.")