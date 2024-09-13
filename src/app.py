# Import Required Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
data_dir = 'data/raw'
csv_file = 'sampleSubmission.csv'
test_csv_file = 'testsampleSubmission.csv'
model_save_path = 'saved_model.h5'
image_size = (224, 224)
batch_size = 32

# Load CSV Data
df = pd.read_csv(csv_file)
df['file_path'] = df['id'].apply(lambda x: os.path.join(data_dir, f"{x}.jpg"))

# Split Data
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Define Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='auto')

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early]
)

# Plot Training History
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Make Predictions on Test Data
test_df = pd.read_csv(test_csv_file)
test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(data_dir, f"{x}.jpg"))

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='file_path',
    y_col=None,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Load Saved Model
model = load_model(model_save_path)

# Predict
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)

# Save Predictions
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': predicted_labels.flatten()
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


# Helper function to load and preprocess images
def load_images_from_folder(folder, num_images=9):
    """
    Load and preprocess images from a given folder.

    Args:
        folder (str): Path to the folder containing images.
        num_images (int): Number of images to load.

    Returns:
        list: Preprocessed images as numpy arrays.
    """
    image_files = os.listdir(folder)[:num_images]
    images = []
    
    for file_name in image_files:
        try:
            img_path = os.path.join(folder, file_name)
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as needed
            img_array = image.img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")

    return images

# Function to display images in a grid
def display_images(images, title):
    """
    Display a list of images in a grid.

    Args:
        images (list): List of images as numpy arrays.
        title (str): Title for the images.
    """
    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load and display Dog images
dog_images = load_images_from_folder("../data/raw/dog")
display_images(dog_images, "Dog Image")

# Load and display Cat images
cat_images = load_images_from_folder("../data/raw/cat")
display_images(cat_images, "Cat Image")

# Define Image Data Generators for Training and Testing
image_size = (224, 224)
batch_size = 32

datagen_train = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Include validation split
datagen_test = ImageDataGenerator(rescale=1./255)

train_data = datagen_train.flow_from_directory(
    "../data/raw",
    target_size=image_size,
    batch_size=batch_size,
    classes=["dog", "cat"],
    subset='training',
    class_mode='categorical'
)

validation_data = datagen_train.flow_from_directory(
    "../data/raw",
    target_size=image_size,
    batch_size=batch_size,
    classes=["dog", "cat"],
    subset='validation',
    class_mode='categorical'
)

# Model Definition
model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Define Callbacks
checkpoint = ModelCheckpoint("../models/vgg16_1.keras", monitor="val_accuracy", verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor="val_accuracy", patience=3, verbose=1)

# Train the Model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# Plot the Training Results
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Accuracy and Loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()

# Load and Predict on a New Image
img_path = "../data/raw/test/9.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

saved_model = load_model("../models/vgg16_1.keras")
prediction = saved_model.predict(img_array)

if prediction[0][0] > prediction[0][1]:
    print("Predicted: Cat")
else:
    print("Predicted: Dog")
