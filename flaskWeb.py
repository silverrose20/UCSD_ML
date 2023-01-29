from flask import Flask, render_template, Response
import os
import cv2
import dlib
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Flatten, Layer, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    # Preprocessing steps - resizing the image to be 150x150
    img = tf.image.resize(img, (150, 150))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return iimage
    return img


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Build Distance Layer
# Create Siamese L1 distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# Reload model
model = tf.keras.models.load_model('siamesemodel_VG16.h5',
                                   custom_objects={'L1Dist': L1Dist,
                                                   'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


def generate_frames():
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # Create a HOG face detector using the built-in dlib class
            face_detector = dlib.get_frontal_face_detector()

            # Run the HOG face detector on the image data.
            # The result will be the bounding boxes of the faces in our image.
            detected_faces = face_detector(frame, 1)
            number_of_faces = len(detected_faces)

            # Loop through each face found in the image
            if number_of_faces == 2:
                for i, face_rect in enumerate(detected_faces):
                    # Crop image and save
                    crop_image = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
                    cv2.imwrite("face{}.jpg".format(i), crop_image)

                # verify the faces
                face_0 = tf.data.Dataset.list_files('face0.jpg').take(1)
                face_1 = tf.data.Dataset.list_files('face1.jpg').take(1)
                faces = tf.data.Dataset.zip((face_0, face_1, tf.data.Dataset.from_tensor_slices(tf.ones(1))))

                faces = faces.map(preprocess_twin)

                faces = faces.take(1)
                faces = faces.batch(1)

                face_1, face_2, y_true = faces.as_numpy_iterator().next()

                prediction = model.predict([face_1, face_2])
                pred = np.round(prediction * 100, 2)

                # Same person = green box around the faces and "Verified" + percentage on the screen
                if prediction > 0.75:
                    for i, face_rect in enumerate(detected_faces):
                        cv2.rectangle(frame, (face_rect.left(), face_rect.top()),
                                      (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, f'Verified {pred}%', (200, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                    # Different people = red box around the faces and "Not Verified" + percentage on the screen
                else:
                    for i, face_rect in enumerate(detected_faces):
                        cv2.rectangle(frame, (face_rect.left(), face_rect.top()),
                                      (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
                    cv2.putText(frame, f'Not Verified {pred}%', (200, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

