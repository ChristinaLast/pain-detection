import numpy as np
import scipy.io
import tensorflow as tf
import asyncio
import time
import traceback

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
                19, activation="tanh", input_shape=(19,)
            ),  # First layer with 19 neurons
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
    5, 19
)  # Initialize if correct weights not available
second_layer_biases = np.array(b[1, 0]).flatten()

# Transpose second layer weights to match TensorFlow's expected shape
second_layer_weights = second_layer_weights.T  # Transpose to shape (19, 5)

model.layers[0].set_weights([first_layer_weights, first_layer_biases])
model.layers[1].set_weights([second_layer_weights, second_layer_biases])

# Prepare your input data (replace this with your actual input data)
text_stream = [
    "Hello there! ",
    "I'm very excited to show ",
    "you a demonstration of text streaming ",
    "using the Hume expressive communication platform.",
]


async def main():
    try:
        client = HumeStreamClient("149BtIPH27m8pyrZDYPzcB2GKkbBgDUGD92mq38ftqGIHYSj")
        config = LanguageConfig(granularity="sentence")
        async with client.connect([config]) as socket:
            for text_sample in text_stream:
                # Simulate real time speaking with a delay
                time.sleep(0.25 * len(text_sample.split(" ")))
                result = await socket.send_text(text_sample)
                print(result)
                emotions = result["language"]["predictions"][0]["emotions"]

                # Make a prediction
                prediction = model.predict(emotions)

                # Print the prediction
                print("Prediction:", prediction)

                print(f"\n{text_sample}")
                # print_emotions(emotions)
    except Exception:
        print(traceback.format_exc())


# When running the streaming API outside of a Jupyter notebook you do not need these lines.
# Jupyter has its own async event loop, so this merges main into the Jupyter event loop.
# To run this sample in a script with asyncio you can use `asyncio.run(main())`
loop = asyncio.get_event_loop()
loop.create_task(main())