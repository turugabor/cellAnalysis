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
import os
import re
from tqdm.notebook import tqdm
from skimage import filters

class Image:
    """basic image class which stores the image metadata:
            - image location
            - image type
            - channel info
            - etc."""

<<<<<<< HEAD
    def __init__(self, file_path):
        # or __post_init__
        # definiáljuk a self.image_path paramétert!
        # self.channel_number
        # self.nucleus_channel 
           self.image_path = Path(file_path)

  
=======
    def __init__(self, folder, name, channel_names = {1:'1', 2:'2', 3:'3'}):
        """
        Description of __init__

        Args:
            folder (str): folder location of the image
            name (str): name of the image
            channel_names (dict): names of the channels, defaults to {1:'1',2:'2',3:'3'}
>>>>>>> 2f3f4b7cdd2e08073fcce05903482cb56e06b96a

        """
        
        self.folder = folder
        self.name = name
        self.channel_names = channel_names
        

    def load_image(self):
        """loads the image data and stores it in self.image"""
        """loads the image data and stores it in self.image"""
<<<<<<< HEAD
 
        if 'tif' in self.image_path[0]:
            self.image = io.imread(*self.image_path)
        elif 'czi' in self.image_path[0]:
            self.image = AICSImage(*self.image_path).get_image_data()
        
        

    def display_image(self):
        """displays the image channels on a pyplot figure"""
        plt.imshow(self.image, cmap='gray')
        plt.show
        

    def set_background(self):
        """int or array"""
        pass

=======

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

>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
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
        masks, __, __, __ = self.cell_model.eval(img, channels=channels, diameter=diameter, **kwargs)
        return masks

    def predict_nuclei(self, img, nucleus_channel=1, diameter=100, **kwargs):
        """...."""
        masks, __, __, __ = self.nucleus_model.eval(img, channels=[nucleus_channel, 0], diameter=diameter, **kwargs)
        return masks


# -

class Experiment:
    """top class of the package"""

    
    def __init__(self, detector, folder, channel_names = {1:'1', 2:'2', 3:'3'}, channel_to_analyse = 2):
        self.folder = folder
        self.detector = detector
        self.channel_names = channel_names
        self.channel_to_analyse = channel_to_analyse
        self.analyzer = Analyzer(self.detector, self.channel_to_analyse)
        self.background = 0

    def get_images(self):
        """collects all the images belonging to the experiment (Image classes)"""
        ## should create a list/dictionary/tuple of images
        # img.load_image()
        # should be defined for the individual experiment
        pass

    def analyse_experiment(self):
        results = []
        for img in tqdm(self.images):
            self.images[img].load_image()
            df = self.analyzer.analyze(self.images[img].image, self.background, self.channel_to_analyse)
            df["name"] = img
            results.append(df)
            del self.images[img].image
        return pd.concat(results)

    def display_cells(self, cellIds):
        """displays a collection of cells by IDs in a single plot
            
            Useful for checking a collection a cells with same properties (e.g. size, fluorescence, etc.)"""
        pass

class ImageXpressExperiment(Experiment):
    
    def __init__(self, detector, folder, channel_names = {1:'1', 2:'2', 3:'3'}, channel_to_analyse = 2, no_bckgrd_imgs = 100):
        super().__init__(detector, folder, channel_names, channel_to_analyse)
        self.no_bckgrd_imgs = no_bckgrd_imgs
        self.get_images()
        self.get_background()
        

    def get_images(self):
        files = os.listdir(self.folder)
        files = [f for f in files if 'thumb' not in f]
        names = [re.findall('_[A-Z][0-9][0-9]_[a-z][0-9]*_',f)[0][1:-1] for f in files]
        names = list(set(names))
        self.images = {}
        for name in names:
            self.images[name] = ImageXpressImage(self.folder, name, self.channel_names)

    def get_background(self):
        stack = []
        imgs = np.random.choice(list(self.images.keys()), self.no_bckgrd_imgs)
        print("calculating background....")
        for name in tqdm(imgs):
            self.images[name].load_image()
            img = self.images[name].image
            img = np.expand_dims(img, 3).copy()
            del self.images[name].image
            stack.append(img)
        stack = np.concatenate(stack, axis =3)
        self.background  = np.median(stack, axis=3).astype(np.uint16)
        # blur the image to further remove noise
        self.background = filters.gaussian(self.background,
                                   sigma=(100, 100),
                                   truncate=10,
                                   multichannel=True, preserve_range=True)        



<<<<<<< HEAD






=======
class Analyzer:
    
    def __init__(self, detector, channel_to_analyse):
        
        self.detector = detector

<<<<<<< HEAD
    def analyze(self, img, nuclear_mask = None, cell_mask = None):
        nucleus_img = img[:, :, 2]
        joint = np.concatenate([nuclear_mask.reshape(-1,1) , nucleus_img.reshape(-1,1)], axis=1)
        data = pd.DataFrame(joint, columns=['cell_idx', 'fluorescence'])
        data = data[data.cell_idx > 0]
        nuc_siz = data.pivot_table(columns=['cell_idx'], aggfunc='size')
        nuc_siz = pd.DataFrame(nuc_siz, columns=['nucleus_size'])
        nuc_flu=data.groupby('cell_idx').mean().rename(columns={"fluorescence": "nucleus_fluoreescence"})
        data = pd.concat([nuc_siz,nuc_flu], axis=1)
        return data # df
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
=======
    def analyze(self, img, background, channel_to_analyse):
        # remove background first
        img = img.astype(int) - background
        img[img < 0] = 0
        img = img.astype(np.uint16)

        nuclei = self.detector.predict_nuclei(img)
        cells = self.detector.predict_cells(img)
        img = img[:,:,channel_to_analyse-1].reshape(-1,1)
        nuclei = (nuclei.reshape(-1,1) > 0).astype(int) # convert the mask ids to ones
        cells = cells.reshape(-1,1)
        cell_fluorescence = (cells>0) * img 
        nucleus_fluorescence = nuclei * img

        data = np.concatenate((cells, img, nuclei, nucleus_fluorescence), axis = 1)
        data = pd.DataFrame(data, columns = ['cell_id', 'cell_fluorescence', 'nucleus_size', 'nucleus_fluorescence'])
        data['cell_size'] = 1
        
        data = data.groupby('cell_id').sum().reset_index()
        data = data[data.cell_id > 0] # drop the background

        return data
>>>>>>> 2f3f4b7cdd2e08073fcce05903482cb56e06b96a
