import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('face_video.avi', fourcc, 20.0, (640, 480))  # Save video at 640x480 resolution

print("Recording video... Press 'q' to stop recording.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow("Recording Face", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as 'face_video.avi'.")