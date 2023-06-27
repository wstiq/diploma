import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Load the saved model
model = load_model('skin_disease_30epochs.h5')

# Define the input image dimensions and batch size
image_width = 64
image_height = 64
batch_size = 32

# Load and preprocess your test data
test_data_dir = 'C:/skinphotos/z_validate'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test data
test_results = model.evaluate(test_generator, verbose=0)
test_loss = test_results[0]
predicted_labels = np.argmax(model.predict(test_generator), axis=1)
true_labels = test_generator.classes

print('Test Loss:', test_loss)

# Calculate precision, recall, and F1-score
report = classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices)
print('Classification Report:')
print(report)
