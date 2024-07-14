import face_recognition
import cv2
import pickle
import os

# Locate the Haar Cascade XML file for face detection
haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')

# Initialize the Haar Cascade face detector
face_detector = cv2.CascadeClassifier(haar_cascade_path)

# Load precomputed face encodings and their associated names
with open('face_enc', 'rb') as file:
    known_faces = pickle.load(file)

print("Starting video stream...")
video_stream = cv2.VideoCapture(0)

# Process each frame from the video stream
while True:
    # Capture frame-by-frame from the video stream
    ret, frame = video_stream.read()
    
    # Convert the frame to grayscale for face detection
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    detected_faces = face_detector.detectMultiScale(
        grayscale_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Compute facial embeddings for each detected face
    face_encodings = face_recognition.face_encodings(rgb_frame)
    identified_names = []

    # Compare each facial encoding with known encodings
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces["encodings"], encoding)
        name = "Unknown"
        
        # If a match is found, determine the most frequent name in the matches
        if True in matches:
            matched_indices = [i for (i, match) in enumerate(matches) if match]
            name_counts = {}

            for index in matched_indices:
                matched_name = known_faces["names"][index]
                name_counts[matched_name] = name_counts.get(matched_name, 0) + 1

            name = max(name_counts, key=name_counts.get)
        
        identified_names.append(name)

    # Annotate the frame with the identified names
    for ((x, y, w, h), name) in zip(detected_faces, identified_names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
video_stream.release()
cv2.destroyAllWindows()
