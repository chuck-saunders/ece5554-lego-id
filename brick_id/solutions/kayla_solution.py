from .solution import Solution
from brick_id.dataset.catalog import Brick
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import numpy as np
import os


class KaylaSolution(Solution):
    def __init__(self, **kwargs):
        super().__init__()
        cwd = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(cwd, 'class_names.txt', 'r')) as f:
            self.class_names = [line.strip() for line in f]

        self.model = tf.keras.models.load_model(os.path.join(cwd, '10_model.keras'))
        self.img_height = 64
        self.img_width = 64

    def identify(self, blob):
        img = blob.resize(self.img_width, self.img_height)

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )
        # TODO: Write your implementation
        return Brick.NOT_IN_CATALOG
