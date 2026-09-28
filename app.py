from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import mediapipe as mp
from PIL import Image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('sign_language_model_with_mediapipe_cnn.h5')

# Assuming these are the classes in order (change as per your dataset)
classes = {
    0: 'ક', 1: 'ખ', 2: 'ગ', 3: 'ઘ', 4: 'ચ', 5: 'છ',
    6: 'જ', 7: 'ઝ', 8: 'ટ', 9: 'ઠ', 10: 'ડ', 11: 'ઢ',
    12: 'ણ', 13: 'ત', 14: 'થ', 15: 'દ', 16: 'ધ', 17: 'ન',
    18: 'પ', 19: 'ફ', 20: 'બ', 21: 'ભ', 22: 'મ', 23: 'ય',
    24: 'ર', 25: 'લ', 26: 'વ', 27: 'શ', 28: 'ષ', 29: 'સ',
    30: 'હ', 31: 'ળ', 32: 'ક્ષ',33:'જ્ઞ'
}  # Update this with actual signs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ===== MediaPipe preprocessing =====
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Get bounding box of first detected hand
            h, w, _ = img.shape
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Crop and resize
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)
            cropped_hand = img[y_min:y_max, x_min:x_max]
            resized_img = cv2.resize(cropped_hand, (64, 64))  # Adjust size if needed
            input_img = resized_img / 255.0  # Normalize
            input_img = np.expand_dims(input_img, axis=0)

            # Predict
            predictions = model.predict(input_img)
            predicted_class = classes[np.argmax(predictions)]
        else:
            predicted_class = "No hand detected"

        return render_template('index.html', prediction=predicted_class, filepath=filepath)

if __name__ == '__main__':
    app.run(debug=True)
