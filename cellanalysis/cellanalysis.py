# +
from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from skimage import io # conda install scikit-image
from aicsimageio import AICSImage  # pip install aicsimageio és pip install aicspylibczi
from pathlib import Path
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_closing
from skimage import measure


class Image:
    """basic image class which stores the image metadata:
            - image location
            - image type
            - channel info
            - etc."""

    def __init__(self, file_path):
        # or __post_init__
        # definiáljuk a self.image_path paramétert!
        # self.channel_number
        # self.nucleus_channel 
        self.image_path = Path(file_path)

        # self.ext = self.image_path.suffix
        # if self.ext not in ['.tif', '.czi']:
        #     raise ValueError(f"Extention '{self.ext}' is not supported!")


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
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def subtract_background(self):
        median = np.median(self.image, axis=[0,1])
        self.image = self.image - median
        self.image = np.clip(self.image, 0, 4095)


class ZeissCziImage(Image):
    
    def __init__(self, folder, name):
        self.image_name = Path(name)
        if self.image_name.suffix == "":
            self.image_name = self.image_name.with_suffix('.czi')
        elif self.image_name.suffix != ".czi":
            raise ValueError("Only supports images with .czi extension")
            
        super().__init__(folder)

    
    def load_image(self):
        img = self.image_path / self.image_name
        aics_image = AICSImage(img)
        self.image = aics_image.get_image_data('YXZ')

class ImageXpressImage(Image):
    
    def __init__(self, folder, name):
        self.image_name = name
        super().__init__(folder)
        
        

    def load_image(self):
        imgs = listdir(self.image_path)
        imgs = [im for im in imgs if self.image_name in im & "_thumb" not in im]
        
        imgs = [io.imread(f"{self.image_path}/{im}") for im in imgs]
        imgs = [im.reshape(im.shape[0], im.shape[1], 1) for im in imgs]
        self.image = np.concatenate(imgs, axis=2)


# -

class CellDetector:
    """class with cell and nucleus detecting methods and detector specific attributes"""
    
    def __init__(self):
        pass

    def predict_cells(self):
        pass

    def predict_nuclei(self, img):
        global_thresh = threshold_otsu(img[:,:,0])
        binary_global = img[:,:,0] > global_thresh
        binary_closed = binary_closing(binary_global, np.ones(shape=(10,10)))
        blobs_labels = measure.label(binary_closed, background=0)
        return blobs_labels

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
    
    def __init__(self):
        pass
    
    def analyze(self, img, nuclear_mask = None, cell_mask = None):
        return data # df
