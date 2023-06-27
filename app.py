import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('skin_disease_best.h5')

# Define the disease labels
disease_labels = ['Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Solar Lentigo', 'Squamous Cell Carcinoma', 'Vascular Lesion']

# Define the threshold for Nevus classification
threshold = 82.0


# Define routes and their respective functions
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the 'photo' file has been submitted
        if 'photo' not in request.files:
            return render_template('upload.html', error='No photo file selected.')

        photo = request.files['photo']

        # Check if the file is empty
        if photo.filename == '':
            return render_template('upload.html', error='No photo file selected.')

        # Save the photo to a temporary directory
        photo_path = os.path.join(app.root_path, 'temp', photo.filename)
        photo.save(photo_path)

        # Preprocess the photo
        img = image.load_img(photo_path, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make prediction using the loaded model
        prediction = model.predict(img)
        disease_index = np.argmax(prediction)
        predicted_disease = disease_labels[disease_index]
        probability = prediction[0][disease_index] * 100

        # Check if the predicted disease is Nevus and the probability is below the threshold
        if predicted_disease == 'Nevus' and probability < threshold:
            predicted_disease = 'Potential Melanoma'

        # Remove the temporary photo file
        os.remove(photo_path)

        return render_template('upload.html', prediction=predicted_disease, probability=probability)

    return render_template('upload.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
