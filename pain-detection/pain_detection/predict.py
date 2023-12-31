import numpy as np
import tensorflow as tf
import asyncio
import cv2
import mediapipe as mp
from hume import HumeStreamClient
from hume.models.config import LanguageConfig
import os
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific scikit-learn warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')




# Loading with pickle
with open('models/scaler_100.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Disable GPU usage by TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Check if TensorFlow is using GPU or CPU
def check_tf_device():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("TensorFlow is using CPU")

check_tf_device()

# Load the trained model
model = tf.keras.models.load_model("models/your_model_100.h5")


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)


# Function to extract features
def extract_features(landmarks):
    features = {}
    # Example: Extracting nose coordinates (replace with correct landmark index)
    nose = landmarks.landmark[1]
    features["NOSE_X"] = nose.x
    features["NOSE_Y"] = nose.y
    features["NOSE_Z"] = nose.z
    # Head rotation (simplified example, actual calculation is more complex)
    # Placeholder values used, replace with correct landmark indices
    head_rotation = landmarks.landmark[2]
    features["HEAD_ROTATION_X"] = head_rotation.x
    features["HEAD_ROTATION_Y"] = head_rotation.y
    features["HEAD_ROTATION_Z"] = head_rotation.z
    # Mouth openness
    # Placeholder values used, replace with correct landmark indices
    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]
    features["MOUTH_VERTICAL"] = lower_lip.y - upper_lip.y
    # Add logic for other features here
    return features


def preprocess_features(features):
    # Convert to a list in the correct order
    feature_list = [features[key] for key in sorted(features)]
    
    # Convert to a numpy array and reshape to 2D
    feature_array = np.array([feature_list])  # Notice the extra brackets

    # Scale the features
    scaled_features = scaler.transform(feature_array)
    return scaled_features



async def main():
    try:
        client = HumeStreamClient("149BtIPH27m8pyrZDYPzcB2GKkbBgDUGD92mq38ftqGIHYSj")
        config = LanguageConfig(granularity="sentence")
        async with client.connect([config]) as socket:
            # Read the video
            cap = cv2.VideoCapture("videos/man in pain clip.mp4")
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # Convert the BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                # Draw the face mesh annotations on the image
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract features for each frame
                        frame_features = extract_features(face_landmarks)

                        # Preprocess the features to match the input shape of the model
                        processed_features = preprocess_features(frame_features)
                    
                        # Make a prediction
                        prediction = model.predict(processed_features)
                        print("Prediction:", prediction)
                        mp.solutions.drawing_utils.draw_landmarks(
                            image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
                        )
                # Display the image (optional)
                # cv2.imshow(
                #     "MediaPipe Face Mesh", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # )
            #     if cv2.waitKey(5) & 0xFF == 27:
            #         break
            # cap.release()
            # cv2.destroyAllWindows()
            # Make a prediction

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the main function in the event loop
asyncio.run(main())
