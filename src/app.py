# -*- coding: utf-8 -*-


from flask import Flask, render_template, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os
import csv

app = Flask(__name__)

model = load_model('model/model.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

from flask import redirect, url_for

import base64

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        # Create 'uploads' folder if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Check if the file part is present in the request
        if 'file' not in request.files:
            return render_template('test.html', error='No file part')

        # Get the file from the request
        file = request.files['file']

        # Check if the file has a valid filename
        if file.filename == '':
            return render_template('test.html', error='No selected file')

        # Save the file with a secure filename to the 'uploads' folder
        file_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(file_path)

        # Process the uploaded image with your brain tumor detection model
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the loaded model
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        result = classes[class_index]

        # Encode the image into base64 format for display
        _, img_encoded = cv2.imencode('.png', img)
        img_base64 = 'data:image/png;base64,' + base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # Print intermediate results for debugging
        print("Raw Prediction:", prediction)
        print("Class Index:", class_index)
        print("Predicted Class:", result)

        # Pass the result, filename, and base64 encoded image to the template
        return render_template('result.html', filename=file.filename, result=result, img_base64=img_base64)

    return render_template('test.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/contact', methods=['POST'])
def submit_contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    # Save form data to a CSV file
    csv_file = 'contact.csv'
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Name', 'Email', 'Message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'Name': name, 'Email': email, 'Message': message})

    return redirect(url_for('contact'))

if __name__ == '__main__':
    app.run(debug=True)

