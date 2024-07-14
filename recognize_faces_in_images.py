import os
import cv2
import pickle
import face_recognition

# Locate the path of the Haar Cascade XML file for face detection
haar_cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_alt2.xml')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load the known faces and their encodings from a previously saved file
with open('face_enc', 'rb') as file:
    known_data = pickle.load(file)

# Load the image where you want to detect faces
image_path = r'C:\Users\ragha\OneDrive\Desktop\Face_Recognition_using_OpenCV\Images\amole_gupte\1c3cf3fff1.jpg'
image = cv2.imread(image_path)

# Convert the image from BGR to RGB (used by face_recognition)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale for Haar Cascade detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
detected_faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Get the facial embeddings for the detected faces
face_encodings = face_recognition.face_encodings(rgb_image)

# Initialize a list to store the names of recognized faces
recognized_names = []

# Iterate over the facial embeddings
for encoding in face_encodings:
    # Compare the detected encodings with known encodings
    matches = face_recognition.compare_faces(known_data["encodings"], encoding)
    
    # Default name is "Unknown" if no match is found
    name = "Unknown"

    # Check if any matches were found
    if True in matches:
        # Get the indices of all matched faces
        matched_indices = [i for i, match in enumerate(matches) if match]
        name_counts = {}

        # Count the occurrences of each matched name
        for index in matched_indices:
            matched_name = known_data["names"][index]
            name_counts[matched_name] = name_counts.get(matched_name, 0) + 1

        # Determine the name with the highest count
        name = max(name_counts, key=name_counts.get)

    # Append the recognized name to the list
    recognized_names.append(name)

# Draw rectangles and names around the detected faces
for ((x, y, w, h), name) in zip(detected_faces, recognized_names):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Display the image with the detected and recognized faces
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
