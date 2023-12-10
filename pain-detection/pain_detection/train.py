import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "datasets/extracted_video_features_binary.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Selecting the columns for features and target
X = df.iloc[:, 5:13]  # Columns 6 to 13 as features
y = df.iloc[:, 4]  # Column 5 as target

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            128, activation="relu", input_shape=(X_train_scaled.shape[1],)
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(
            1, activation="sigmoid"
        ),  # Assuming binary classification
    ]
)

# Compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=10,
    batch_size=32,
)

# You can save the model for later use
model.save("your_model.h5")
