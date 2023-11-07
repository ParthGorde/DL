import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to your animal dataset directory
animal_data_dir = 'C:\\Users\\User\\Desktop\\A'


input_shape = (128, 128, 3)  # Update this based on your image size and channels
num_classes = 10  # Update this to the number of classes in your animal dataset

# Initialize ImageDataGenerator for preprocessing and augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalizing pixel values

# Load and preprocess the animal dataset using ImageDataGenerator
batch_size = 32
train_generator = datagen.flow_from_directory(
    animal_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    animal_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

animal_data_dir = 'C:\\Users\\User\\Desktop\\A\\class.csv'
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print("Test Accuracy: {:.2f}%".format(test_acc * 100))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()