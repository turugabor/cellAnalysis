import numpy as np
import pandas as pd
from skimage import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import os
import logging

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# -

from skimage.measure import label

TRAIN_LENGTH = 448
BATCH_SIZE = 16
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 2
EPOCHS = 200
VALIDATION_STEPS = 15
SEED = 99


class ImageGenerator:
    def __init__(self, folder):
        self.folder = folder
        (
            image_data_generator,
            mask_data_generator,
            test_image_data_generator,
            test_mask_data_generator,
        ) = self.create_generators()
        self.train_generator = self.generate_image_mask_weights(
            image_data_generator, mask_data_generator
        )
        self.test_generator = self.generate_image_mask_weights(
            test_image_data_generator, test_mask_data_generator
        )
        

    def generate_image_mask_weights(self, image_data_generator, mask_data_generator):
        train_generator = zip(image_data_generator, mask_data_generator)
        for (img, mask) in train_generator:
            # The weights for each class, with the constraint that:
            #     sum(class_weights) == 1.0
            class_weights = tf.constant([0.4, 0.6])
            class_weights = class_weights / tf.reduce_sum(class_weights)

            # Create an image of `sample_weights` by using the label at each pixel as an
            # index into the `class weights` .
            sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))
            yield (img, mask, sample_weights)

    def convert_to_mask(self, img):
        y = tf.constant([0])
        img = tf.math.greater(img, y)
        return  tf.cast(img, tf.float32)

    # define the ImageDataGenerator instances with augmentation
    def create_generators(self):
        print(Path(self.folder) / Path("train/images"))
        image_data_generator = ImageDataGenerator(
            rescale=1.0 / 4095.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="constant",
            cval=0,
            rotation_range=30,
        ).flow_from_directory(
            Path(self.folder) / Path("train/images"),
            batch_size=16,
            target_size=(256, 256),
            class_mode=None,
            classes=["images"],
            seed=SEED,
        )

        mask_data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=30,
            fill_mode="constant",
            cval=0,
            preprocessing_function=self.convert_to_mask,
        ).flow_from_directory(
            Path(self.folder) / Path("train/masks"),
            batch_size=16,
            target_size=(256, 256),
            class_mode=None,
            classes=["masks"],
            color_mode="grayscale",
            seed=SEED,
        )

        test_image_data_generator = ImageDataGenerator(
            rescale=1.0 / 4095.0
        ).flow_from_directory(
            Path(self.folder) / Path("test/images"),
            batch_size=16,
            target_size=(256, 256),
            class_mode=None,
            classes=["images"],
            seed=SEED,
        )

        test_mask_data_generator = ImageDataGenerator(
            preprocessing_function=self.convert_to_mask
        ).flow_from_directory(
            Path(self.folder) / Path("test/masks"),
            batch_size=16,
            target_size=(256, 256),
            class_mode=None,
            classes=["masks"],
            color_mode="grayscale",
            seed=SEED,
        )
        return (
            image_data_generator,
            mask_data_generator,
            test_image_data_generator,
            test_mask_data_generator,
        )


class Vesicles:
    def __init__(self, project_folder=Path(".")):

        self.project_folder = project_folder
        self.get_base_model()
        self.get_stacks()

        self.model = self.unetModel(OUTPUT_CHANNELS)
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.Generator = ImageGenerator(Path(self.project_folder) / Path("labeled_data"))

    def get_base_model(self):
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=[256, 256, 3], include_top=False
        )

        # Use the activations of these layers
        layer_names = [
            "block_1_expand_relu",  # 64x64
            "block_3_expand_relu",  # 32x32
            "block_6_expand_relu",  # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",  # 4x4
        ]
        self.base_model_outputs = [
            self.base_model.get_layer(name).output for name in layer_names
        ]

    def get_stacks(self):
        self.down_stack = tf.keras.Model(
            inputs=self.base_model.input, outputs=self.base_model_outputs
        )
        self.down_stack.trainable = False

        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
            pix2pix.upsample(32, 3),  # 64x64 -> 128x128
        ]

    def unetModel(self, output_channels):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        # Downsampling through the model
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2, padding="same"
        )

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def plot_model(self):
        return tf.keras.utils.plot_model(self.model, show_shapes=True)

    def createMask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask

    def show_predictions(self, test=True):
        if test:
            generator = self.Generator.test_generator
        else:
            generator = self.Generator.train_generator

        r = np.random.randint(30)
        is_there_mask = False

        while (
            not is_there_mask
        ):  # skip the images without vesicles for displazing purposes
            a = next(generator)
            image, mask = a[0], a[1]
            is_there_mask = a[1][0].sum() > 10
        pred_mask = self.model.predict(image)
        self.utils.display([image[0], mask[0], self.createMask(pred_mask)[0]])

    def train(self):
        early_stop = EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=2
        )

        self.model_history = self.model.fit(
            self.Generator.train_generator,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            validation_data=self.Generator.test_generator,
            use_multiprocessing=False,
            callbacks=[early_stop],
        )

    def plot_training(self):
        loss = self.model_history.history["loss"]
        val_loss = self.model_history.history["val_loss"]

        plt.figure()
        plt.plot(self.model_history.epoch, loss, "r", label="Training loss")
        plt.plot(self.model_history.epoch, val_loss, "bo", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        # plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def load_model(self, name = 'model'):
        self.model.load_weights(Path(self.project_folder) / Path(f"models/{name}"))

    
    def save_model(self, name = 'model'):
        self.model.save_weights(Path(self.project_folder) / Path(f"models/{name}"))

