from .solution import Solution
from brick_id.dataset.catalog import Brick, allowable_parts
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import numpy as np
import os
import cv2


class KaylaSolution(Solution):
    def __init__(self, **kwargs):
        super().__init__()
        cwd = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(cwd, 'class_names.txt'), 'r') as f:
            self.class_names = [line.strip() for line in f]

        model_path = os.path.join(cwd, 'b10_model.keras')
        if not os.path.exists(model_path):
            print(f'Failed to find the model at {model_path}!')
            raise FileNotFoundError

        self.model = tf.keras.models.load_model(model_path)
        self.img_height = 64
        self.img_width = 64
        self.catalog = allowable_parts()

    def identify(self, blob: np.ndarray) -> Brick:
        img = cv2.resize(blob, (self.img_width, self.img_height))

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        brick_id = self.class_names[np.argmax(score)]
        brick_description = self.catalog[brick_id]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence. Description: {}"
            .format(brick_id, 100 * np.max(score), brick_description)
        )
        # TODO: Write your implementation
        return Brick['_' + brick_id]
