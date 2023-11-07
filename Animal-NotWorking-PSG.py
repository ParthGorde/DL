import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
zoo_data = pd.read_csv("zoo.csv")
class_data = pd.read_csv("class.csv")

# Merge data
combined_data = pd.concat([zoo_data, class_data], axis=0)

# Extract features
x_image_data = combined_data.iloc[:, 1:17].values

# Reshape for convolution
x_image_data = x_image_data.reshape(-1, 4, 4, 1)

# Extract labels
y_data = combined_data["Class_Number"].values

# Get number of classes
num_classes = len(np.unique(y_data))
print("Number of Classes:", num_classes)

# Get unique class names
class_names = combined_data['Class_Number'].unique()

# Map class names to integers
class_mapping = {name:index for index, name in enumerate(class_names)}

# Convert class names to integers
y_data = combined_data['Class_Number'].map(class_mapping)

# Encode labels
num_classes = len(class_mapping)
y_data = tf.keras.utils.to_categorical(y_data, num_classes)
# Split data
(x_train, x_test, y_train, y_test) = train_test_split(x_image_data, y_data,
                                                      test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(4, 4, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# Set units to match number of classes
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compile and train
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=10,
                    validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy: {}".format(test_acc))
