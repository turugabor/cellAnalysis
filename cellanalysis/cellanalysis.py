# +
from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from skimage import io # conda install scikit-image
from aicsimageio import AICSImage  # pip install aicsimageio és pip install aicspylibczi
from pathlib import Path
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_closing
from skimage import measure
import pandas as pd
from cellpose import models

class Image:
    """basic image class which stores the image metadata:
            - image location
            - image type
            - channel info
            - etc."""

    def __init__(self, folder, name, channel_names = {1:'1', 2:'2', 3:'3'}):
        """
        Description of __init__

        Args:
            folder (str): folder location of the image
            name (str): name of the image
            channel_names (dict): names of the channels, defaults to {1:'1',2:'2',3:'3'}

        """

        self.folder = folder
        self.name = name
        self.channel_names = channel_names


    def load_image(self):
        """loads the image data and stores it in self.image"""
        """loads the image data and stores it in self.image"""

        pass

    def display_image(self):
        channels = self.image.shape[2]
        plt.figure(figsize=(5*channels,5))
        for channel in range(channels):
            plt.subplot(1,3, channel + 1)
            plt.imshow(self.image[:,:,channel], cmap='gray')
            plt.title(f"channel {channel + 1}: {self.channel_names[channel + 1]}")
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def subtract_background(self):
        median = np.median(self.image, axis=[0,1])
        self.image = self.image - median
        self.image = np.clip(self.image, 0, 4095)


class ZeissCziImage(Image):

    def __init__(self, folder, name, channel_names = {1:'1', 2:'2', 3:'3'}):
        super().__init__(folder, name, channel_names)


    def load_image(self):
        path = f"{self.folder}/{self.name}"
        aics_image = AICSImage(path)
        self.image = aics_image.get_image_data('YXZ')

class ImageXpressImage(Image):

    def __init__(self, folder, name, channel_names = {1:'1', 2:'2', 3:'3'}):
        super().__init__(folder, name, channel_names)

    def load_image(self):
        imgs = listdir(self.folder)
        imgs = [im for im in imgs if f"_{self.name}_" in im] # list comprehension
        imgs = [im for im in imgs if f"_thumb" not in im]
        imgs.sort()
        imgs = [io.imread(f"{self.folder}/{im}") for im in imgs]

        imgs = [im.reshape(im.shape[0], im.shape[1], 1) for im in imgs]
        self.image = np.concatenate(imgs, axis=2)


# +
class CellDetector:
    """class with cell and nucleus detecting methods and detector specific attributes"""

    def __init__(self):
        pass

    def predict_cells(self, img, cell_channel=2):
        cell_channel -= 1
        global_thresh = threshold_otsu(img[:,:,cell_channel])
        binary_global = img[:,:,cell_channel] > global_thresh
        binary_closed = binary_closing(binary_global, np.ones(shape=(10,10)))
        blobs_labels = measure.label(binary_closed, background=0)
        return blobs_labels

    def predict_nuclei(self, img, nucleus_channel=1):
        nucleus_channel -=1
        global_thresh = threshold_otsu(img[:,:,nucleus_channel])
        binary_global = img[:,:,nucleus_channel] > global_thresh
        binary_closed = binary_closing(binary_global, np.ones(shape=(10,10)))
        blobs_labels = measure.label(binary_closed, background=0)
        return blobs_labels

class CellPoseDetector:
    """class with cell and nucleus detecting methods and detector specific attributes"""

    def __init__(self):
        self.cell_model = models.Cellpose(gpu=False, model_type='cyto')
        self.nucleus_model = models.Cellpose(gpu=False, model_type='nuclei')

    def predict_cells(self, img, channels=[3,1], diameter=300, **kwargs):
        """...."""
        masks, flows, styles, diams = self.cell_model.eval(img, channels=channels, diameter=diameter, **kwargs)
        return masks

    def predict_nuclei(self, img, nucleus_channel=1, diameter=300, **kwargs):
        """...."""
        masks, flows, styles, diams = \
            self.nucleus_model.eval(img,
                                    channels=[nucleus_channel, 0],
                                    diameter=diameter,
                                    **kwargs)
        return masks


# -

class Experiment:
    """top class of the package"""

    def __init__(self, folder):
        pass

    def get_images(self):
        """collects all the images belonging to the experiment (Image classes)"""
        ## should create a list/dictionary/tuple of images
        # img.load_image()
        pass

    def analyse_experiment(self):

        pass

    def display_cells(self, cellIds):
        """displays a collection of cells by IDs in a single plot
            
            Useful for checking a collection a cells with same properties (e.g. size, fluorescence, etc.)"""
        pass


class Analyzer:

    def __init__(self, detector):

        self.detector = detector

    def analyze(self, img, nucleus_channel):
        nuclei = self.detector.predict_nuclei(img, nucleus_channel)
        img = img[:,:,nucleus_channel].reshape(-1,1)
        nuclei = nuclei.reshape(-1,1)

        data = np.concatenate((nuclei, img), axis = 1)
        data = pd.DataFrame(data, columns = ['cell_id', 'fluorescence'])
        data['size'] = 1

        data = data.groupby('cell_id').sum()
        data['mean_fluorescence'] = data.fluorescence / data["size"]

        return data
