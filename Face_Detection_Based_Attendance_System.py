import cv2
import streamlit as st
from face_encoding import Face
import os

# Initialize face encoder
custom_encoder = Face()
custom_encoder.load_encoding_images("C:/Users/kunwa/Python/Projects/Face-Based-Attendance-System/Dataset")
unique_names = set()

# Function to process video frames and detect faces
def process_frame(frame):
    face_areas, recognized_names = custom_encoder.detect_known_faces(frame)
    for face_coords, found_name in zip(face_areas, recognized_names):
        t, r, b, l = face_coords[0], face_coords[1], face_coords[2], face_coords[3]
        if found_name not in unique_names:
            unique_names.add(found_name)
            with open("attendance_log.txt", "a") as attendance_log:
                attendance_log.write(found_name + "\n")
        cv2.putText(frame, found_name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 200), 4)
    return frame, recognized_names

# Streamlit app
st.title("Face Detection Based Attendance System")

# Add new user section
st.header("Add New User")
new_user_name = st.text_input("Enter new user's name")

if st.button("Capture and Add User"):
    if new_user_name:
        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            new_user_image_path = os.path.join("C:/Users/kunwa/Python/Projects/Face-Based-Attendance-System/Dataset", f"{new_user_name}.jpg")
            cv2.imwrite(new_user_image_path, frame)
            st.success(f"User {new_user_name} added successfully with captured image!")
            
            # Encode the new user's face
            custom_encoder.load_encoding_images("C:/Users/kunwa/Python/Projects/Face-Based-Attendance-System/Dataset")
        else:
            st.error("Error: Could not capture image from webcam.")
        cap.release()
    else:
        st.error("Please provide a name.")

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    st.error("Error: Could not open video capture device.")
else:
    # Display video feed
    frame_placeholder = st.empty()
    attendance_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from video capture device.")
            break

        frame, recognized_names = process_frame(frame)
        frame_placeholder.image(frame, channels="BGR")
        attendance_placeholder.text(f"Recognized Names: {', '.join(recognized_names)}")

    cap.release()