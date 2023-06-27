import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Set the paths for your dataset
train_data_dir = 'C:\skinphotos\z_train'
validation_data_dir = 'C:\skinphotos\z_validate'

# Set the number of classes and the input size
num_classes = 9
input_shape = (64, 64, 3)

# Set the hyperparameters for training
learning_rate = 0.001
batch_size = 32
epochs = 30

# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,  # Add rotation augmentation
    width_shift_range=0.1,  # Add width shift augmentation
    height_shift_range=0.1,  # Add height shift augmentation
    brightness_range=[0.8, 1.2],  # Add brightness augmentation
    fill_mode='nearest'  # Specify the fill mode for augmentation
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create the data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))  # Add dropout regularization
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    min_lr=0.0001
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[reduce_lr]
)

# Save the trained model
model.save('skin_disease_30epochs.h5')

validation_loss, validation_accuracy = model.evaluate(validation_generator, verbose=1)

print("Validation Loss: {:.4f}".format(validation_loss))
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))