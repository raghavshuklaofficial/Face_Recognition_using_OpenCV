import os
import cv2
import pickle
from imutils import paths
import face_recognition

# Get the list of image file paths in the 'Images' directory
image_paths = list(paths.list_images('Images'))

# Initialize lists to store encodings and names
encodings_list = []
names_list = []

# Iterate through each image path
for image_path in image_paths:
    # Extract the name of the person from the directory structure
    name = os.path.basename(os.path.dirname(image_path))
    
    # Load the image and convert it from BGR (used by OpenCV) to RGB
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect the face locations in the image
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    
    # Compute the facial encodings for the detected faces
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Store each encoding and corresponding name
    for encoding in face_encodings:
        encodings_list.append(encoding)
        names_list.append(name)

# Create a dictionary with encodings and names
data = {"encodings": encodings_list, "names": names_list}

# Save the data to a file using pickle
with open("face_enc", "wb") as file:
    pickle.dump(data, file)