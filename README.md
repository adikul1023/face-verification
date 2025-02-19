Face Verification Tool
This is a face verification tool designed to detect faces in real-time, confirm liveness, and differentiate between live faces and spoofing attempts (e.g., photos or videos). The system uses advanced techniques such as head pose estimation, perspective distortion analysis, depth estimation, blink detection, and motion parallax to ensure robust liveness checks.

Features
Face Detection : Detects faces using the face_recognition library.
Liveness Checks :
Head Pose Estimation: Detects head movement to confirm liveness.
Perspective Distortion: Analyzes facial landmarks to detect flat surfaces (e.g., photos).
Depth Estimation: Uses MiDaS to calculate depth maps and identify spoofing attempts.
Blink Detection: Detects eye blinks to confirm a live face.
Motion Parallax: Tracks facial landmarks across frames to detect motion.
Real-Time Processing : Runs on a webcam feed for seamless interaction.
Libraries and Dependencies
The following Python libraries are required to run this project. Install them using the commands below:

bash
Copy
1
2
3
4
5
pip install opencv-python
pip install face-recognition
pip install dlib
pip install torch torchvision
pip install numpy
Additionally, the following pre-trained models are included in the repository:

shape_predictor_68_face_landmarks.dat: A pre-trained model for facial landmark detection (already included in the repo).
File Structure
The repository contains the following main files:

detection.py : Records a video of your face for training.
modeltrain.py : Trains the model using the recorded video to generate face encodings and average depths.
realtimerecognition.py : Performs real-time face recognition and liveness checks.
Steps to Use This Repository
Step 1: Record Your Face
Run the detection.py script to record a video of your face. Follow these steps:

Open a terminal and navigate to the project directory.
Run the following command:
bash
Copy
1
python detection.py
Sit in front of the webcam and move your face slightly to capture different angles. Ensure your face is well-lit and centered.
Press q to stop recording when done. The video will be saved as face_video.avi.
Step 2: Train the Model
Run the modeltrain.py script to process the recorded video and generate face encodings and average depths:

In the terminal, run:
bash
Copy
1
python modeltrain.py
The script will extract face encodings, calculate depth maps, and save the data into a file named face_encodings.pkl.
Step 3: Perform Real-Time Face Verification
Run the realtimerecognition.py script to perform real-time face recognition and liveness checks:

In the terminal, run:
bash
Copy
1
python realtimerecognition.py
The webcam feed will open, and the system will detect your face in real-time.
If your face is recognized and confirmed as live, the system will display a welcome message with your name.
Notes
Lighting Conditions : Ensure proper lighting during both recording and real-time detection for optimal performance.
Webcam Quality : Use a high-quality webcam for better accuracy in face detection and depth estimation.
Confidence Score : The system requires a minimum confidence score of 4 to confirm liveness. You can adjust this threshold in realtimerecognition.py if needed.
Troubleshooting
Error: ImportError or missing libraries :
Ensure all dependencies are installed by running:
bash
Copy
1
pip install -r requirements.txt
Error: FileNotFoundError :
Ensure that shape_predictor_68_face_landmarks.dat and face_encodings.pkl are present in the project directory.
Depth Calculation Issues :
If depth calculation fails frequently, try reducing the resolution of the input frames in realtimerecognition.py.
Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

