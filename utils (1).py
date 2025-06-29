import tensorflow as tf
import numpy as np
from PIL import Image
import json

def load_classifier(classifier_path):
    return tf.keras.models.load_model(classifier_path, compile=False)

def load_label_map(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image_array = np.asarray(image)
    processed_image = process_image(image_array)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [str(i) for i in top_indices]

    return top_probs, top_classes