import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# Extract ZIP dataset
with zipfile.ZipFile('fashion-mnist.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load datasets
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Data preprocessing
X_train, X_test = train_df.drop('label', axis=1).values / 255.0, test_df.drop('label', axis=1).values / 255.0
y_train, y_test = to_categorical(train_df['label'], 10), to_categorical(test_df['label'], 10)

# Model definition
model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}")

# Plot metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
plt.show()

# Visualize a sample
index = 20
plt.imshow(X_train[index].reshape(28, 28), cmap="gray")
plt.title(f"True Label: {np.argmax(y_train[index])}")
plt.show()

# Prediction
prediction = model.predict(X_train[index].reshape(1, 784))
print(f"Predicted Label: {np.argmax(prediction)}")
