import numpy as np
import scipy.io
import tensorflow as tf
import asyncio
import time
import traceback
import cv2
import mediapipe as mp
from hume import HumeStreamClient
from hume.models.config import LanguageConfig


# Load the .mat file
mat = scipy.io.loadmat("models/net1.mat")

# Extract the neural network object
net = mat["net1"][0, 0]

IW = net["IW"]
LW = net["LW"]
b = net["b"]

print(IW[0, 0].shape)  # Should match the input shape and the first layer's units
print(LW[0, 0].shape)  # Should match the first layer's units and second layer's units
print(b[0, 0].shape)  # Should match the first layer's units
print(b[1, 0].shape)  # Should match the second layer's units


# Recreate the model architecture as per MATLAB's model
def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                7, activation="tanh", input_shape=(7,)
            ),  # First layer with 7 neurons
            tf.keras.layers.Dense(
                5, activation="sigmoid"
            ),  # Second layer with 5 neurons
        ]
    )
    return model


model = create_model()

# Assign weights and biases
first_layer_weights = np.array(IW[0, 0])
first_layer_biases = np.array(b[0, 0]).flatten()

# Correct weights for the second layer
second_layer_weights = np.random.rand(
    5, 7
)  # Initialize if correct weights not available
second_layer_biases = np.array(b[1, 0]).flatten()

# Transpose second layer weights to match TensorFlow's expected shape
second_layer_weights = second_layer_weights.T

model.layers[0].set_weights([first_layer_weights, first_layer_biases])
model.layers[1].set_weights([second_layer_weights, second_layer_biases])

# Prepare your input data (replace this with your actual input data)
text_stream = [
    "Hello there! ",
    "I'm very excited to show ",
    "you a demonstration of text streaming ",
    "using the Hume expressive communication platform.",
]


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
    # Assuming 'features' is a dictionary of extracted features
    # Convert to a list in the correct order, matching the input format of the model
    feature_list = [features[key] for key in sorted(features)]
    return np.array([feature_list])  # Convert to a numpy array with shape (1, 7)


async def main():
    try:
        client = HumeStreamClient("149BtIPH27m8pyrZDYPzcB2GKkbBgDUGD92mq38ftqGIHYSj")
        config = LanguageConfig(granularity="sentence")
        async with client.connect([config]) as socket:
            for text_sample in text_stream:
                # Use asyncio.sleep instead of time.sleep
                await asyncio.sleep(0.25 * len(text_sample.split(" ")))
                result = await socket.send_text(
                    text_sample
                )  # Ensure proper use of await
                print(result)
                emotions = result["language"]["predictions"][0]["emotions"]
                print(emotions)

                # Ensure emotions is correctly shaped for the model
                # emotions = np.array(emotions).reshape(1, -1)  # Adjust as necessary

                # Read the video
                cap = cv2.VideoCapture("man in pain clip.mp4")
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
                            print("Prediction:", prediction)
                            print(f"\n{text_sample}")
                            mp.solutions.drawing_utils.draw_landmarks(
                                image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
                            )
                    # Display the image (optional)
                    cv2.imshow(
                        "MediaPipe Face Mesh", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                cap.release()
                cv2.destroyAllWindows()
                # Make a prediction

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


# Run the main function in the event loop
asyncio.run(main())
