import cv2 as cv
import mediapipe as mp
import tempfile
import streamlit as st

mp_face_mesh = mp.solutions.face_mesh

def process_video(uploaded_video, stframe):
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    max_faces = 5
    detection_confidence = 0.5
    tracking_confidence = 0.5

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_video.read())
        video_path = tmpfile.name

    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        st.write("Error: Unable to open video file.")
        return

    with mp_face_mesh.FaceMesh(max_num_faces=max_faces, min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as face_mesh:
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame with MediaPipe
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

            # Update the frame in Streamlit
            if frame_count % 5 == 0:  # Update every 5 frames
                stframe.image(frame, channels="BGR", use_column_width=True)
            frame_count += 1

    video.release()
