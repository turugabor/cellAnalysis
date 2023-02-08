# -*- coding: utf-8 -*-
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
from skimage import measure, exposure
import pandas as pd
from cellpose import models
import os
from tqdm.notebook import tqdm
import re
from skimage import filters
from cellanalysis.pix2pix import Vesicles
import tensorflow as tf
from patchify import patchify, unpatchify
import tensorflow as tf


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
    
    def display_image(self, gamma = 1): 
        channels = self.image.shape[2]
        plt.figure(figsize=(5*channels,5))
        for channel in range(channels):
            plt.subplot(1,3, channel + 1)
            img = exposure.adjust_gamma(self.image[:,:,channel], gamma)
            plt.imshow(img, cmap='gray')
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
        

    def load_image(self, T=0):
        path = f"{self.folder}/{self.name}"
        aics_image = AICSImage(path)
        self.image = aics_image.get_image_data('CYXZ', T=T)
        self.image = self.image.swapaxes(0, 1).swapaxes(1, 2)

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

class Detector:

    def __init__(self):
        pass


class CellDetector(Detector):
    """class with cell and nucleus detecting methods and detector specific attributes"""
    
    def __init__(self):
        super().__init__()

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
    

class CellPoseDetector(Detector):

    """class with cell and nucleus detecting methods and detector specific attributes"""
    
    def __init__(self, channels, cell_diameter=300, nucleus_diameter=70, gpu=True, grayscale=False, custom_model=False, **kwargs):
        """channels:
            # grayscale=0, R=1, G=2, B=3
            # channels = [cytoplasm, nucleus]
            # if NUCLEUS channel does not exist, set the second channel to 0
            # channels = [0,0]"""
        super().__init__()
        
        
        if custom_model:
            self.cell_model = models.CellposeModel(gpu=True, 
                                pretrained_model=custom_model)
        else:
            self.cell_model = models.Cellpose(gpu=gpu, model_type='cyto')    
        
        self.nucleus_model = models.Cellpose(gpu=gpu, model_type='nuclei')
        self.channels = channels
        self.cell_diameter = cell_diameter
        self.nucleus_diameter = nucleus_diameter
        self.grayscale = grayscale
        self.kwargs = kwargs

    def predict_cells(self, img):
        """...."""
        # use model diameter if user diameter is 0
        diameter = self.cell_model.diam_labels if self.cell_diameter==0 else self.cell_diameter
        if not self.grayscale:            
            result = self.cell_model.eval(img, channels=self.channels, diameter=diameter, **self.kwargs)
            
        else:
            cytoplasm_channel = self.channels[0]-1
            result = self.cell_model.eval(img[:,:,cytoplasm_channel], channels=[0,0], diameter=diameter, **self.kwargs)
        
        masks = result[0]
        return masks

    def predict_nuclei(self, img):
        nucleus_channel = self.channels[1]-1
        if not self.grayscale:
            masks, flows, styles, diams = self.nucleus_model.eval(img, channels=[nucleus_channel,0], diameter=self.nucleus_diameter, **self.kwargs)
            
        else:
            masks, flows, styles, diams = self.nucleus_model.eval(img[:,:,nucleus_channel], channels=[0,0], diameter=self.nucleus_diameter, **self.kwargs)
        return masks


class VesicleDetector(Detector):
    """class with cell and nucleus detecting methods and detector specific attributes"""
    
    def __init__(self, vesicle_channel = 0, project_folder = '.', **kwargs):
        super().__init__()
        self.vesiclemodel = Vesicles(project_folder)
        # load the pretrained model or train new
        try:
            self.vesiclemodel.load_model()
            print('Weigths loaded')
        except:
            print(f"Saved weights not found, please train a model (detector.train_model) first or get pretrained weights")

        self.kwargs = kwargs


    def train_model(self):
        self.vesiclemodel.train()
        print('Saving the weights')
        self.vesiclemodel.save_model()

    def split_image(self, img, tile_size):
        image_shape = img.shape
        tile_rows = np.reshape(img, [image_shape[0], -1, tile_size[1], image_shape[2]])
        serial_tiles = np.transpose(tile_rows, [1, 0, 2, 3])
        return np.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

    def pad_image_to_tile_multiple(self, img, tile_size, padding="constant"):
        imagesize = np.array(img.shape[0:2])
        padding_ = np.ceil(imagesize / tile_size).astype(np.int32) * tile_size - imagesize
        return np.pad(img, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)

    def unsplit_image(self, tiles, image_shape):
        tile_width = tiles.shape[1]
        serialized_tiles = np.reshape(tiles, [-1, image_shape[0], tile_width, image_shape[2]])
        rowwise_tiles = np.transpose(serialized_tiles, [1, 0, 2, 3])
        return np.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])

    def create_mask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask
                
    def predict_vesicles(self, img):
        """...."""
        img_dimensions = img.shape

        # adjust exposure for better detection of low contrast images
        # img = exposure.rescale_intensity(img * 1.0)

        # create 3 dimensional img 
        img = np.expand_dims(img, 2)
        img = np.concatenate([img] * 3, axis=2) 

        # pad the image with zerosnincs in case the dimensions are not dividable by 256
        img = self.pad_image_to_tile_multiple(img, 256)
        padded_shape = img.shape
        img = self.split_image(img, (256, 256))  

        pred = self.vesiclemodel.model.predict(img)


        pred = self.unsplit_image(pred, (padded_shape[0], padded_shape[1],2))

        # create mask
        masks = self.create_mask(pred)

        # return the predicted image with the original shape
        return masks[:img_dimensions[0], :img_dimensions[1]]

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
            df["img"] = img
            results.append(df)
            del self.images[img].image
        self.results = pd.concat(results)
        self.results.cell_id = range(len(self.results))

    def load_results(self, path, *args, **kwargs):
        """Loads the analysis results 
        """
        self.results = pd.read_csv(path, *args, **kwargs)
        
    def select_cell(self, idx, size=300):
        
        img_name = self.results[self.results.cell_id == idx]["img"].values[0]
        img = self.images[img_name]
        img.load_image()
        
        # subtract_background
        img.image = img.image.astype(int) - self.background
        img.image[img.image < 0] = 0
        img.image = img.image.astype(np.uint16)
        
        x = self.results[(self.results.cell_id == idx)]['x_coor'].values[0]
        y = self.results[(self.results.cell_id == idx)]['y_coor'].values[0]
        
        return img.image[y-size//2:y+size//2, x-size//2:x+size//2]

    
        

class ArrestinVesicleExperiment(Experiment):
    
    def __init__(self, folder, channel_names={1:'1', 2:'2'}, vesicle_channel=0, cell_channel=1 , no_timepoints=1):
        self.folder = Path(folder)
        self.no_timepoints = no_timepoints
        self.vesicle_channel = vesicle_channel
        self.cell_channel = cell_channel
        
        # create mask folders
        self.mask_folder = folder / Path("masks")
        if not os.path.exists(self.mask_folder):
            os.mkdir(self.mask_folder)

    def detect_cells(self, cell_detector):
        
        czis = os.listdir(self.folder)
        czis = [czi for czi in czis if '.czi' in czi]
        for czi in tqdm(czis):
            name = czi.split(".czi")[0]
            
            for T in tqdm(range(self.no_timepoints)):

                img = ZeissCziImage(self.folder, czi)
                img.load_image(T=T)
                img = img.image[:,:,:, 0] / 4095

                cells = cell_detector.predict_cells(img)
    
                io.imsave(self.mask_folder / Path(f"cells_T{T}_{name}.tif"), cells, check_contrast=False)
        

    def detect_vesicles(self, vesicle_detector):
        czis = os.listdir(self.folder)
        czis = [czi for czi in czis if '.czi' in czi]
        for czi in tqdm(czis):
            name = czi.split(".czi")[0]
            
            for T in tqdm(range(self.no_timepoints)):

                img = ZeissCziImage(self.folder, czi)
                img.load_image(T=T)
                img = img.image[:,:,self.vesicle_channel, 0] / 4095

                vesicles = vesicle_detector.predict_vesicles(img)
                vesicle_idxs = measure.label(np.array(vesicles).astype(np.uint8))
    
                io.imsave(self.mask_folder / Path(f"vesicles_T{T}_{name}.tif"), vesicle_idxs, check_contrast=False)
              

    def analyze_image(self, arrestin, cells, cell_idxs, vesicle_idxs, cell_masks, vesicle_masks):
        pixel_data = np.concatenate(
                [
                vesicle_idxs.reshape(-1, 1),
                vesicle_masks.reshape(-1, 1),
                cell_idxs.reshape(-1, 1),
                cell_masks.reshape(-1, 1),
                arrestin.reshape(-1, 1),
                cells.reshape(-1, 1)
                ],
                axis=1
                )
        pixel_data = pd.DataFrame(pixel_data)
        pixel_data.columns = ['vesicle_index', 'vesicle_mask', 'cell_index',
                        'cell_mask', 'arrestin_fluo', 'L10_fluo']


        vesicle_mean = pixel_data.groupby(['cell_index', 'vesicle_index']).mean()

        arrestin_background = pixel_data.groupby(['cell_index']).mean().loc[0].arrestin_fluo
        cell_background = pixel_data.groupby(['cell_index']).mean().loc[0].L10_fluo

        pixel_data.arrestin_fluo -= arrestin_background
        pixel_data.L10_fluo -= cell_background

        pixel_data.arrestin_fluo.clip(lower=0, inplace=True)

        vesicle_mean = pixel_data.groupby(['cell_index', 'vesicle_index']).mean().reset_index()

        vesicle_sum = pixel_data.groupby(['cell_index', 'vesicle_index']).sum().reset_index()

        # vesicle_mean = vesicle_mean[vesicle_mean.vesicle_index > 0]
        # vesicle_sum = vesicle_sum[vesicle_sum.vesicle_index > 0]

        cell_mean = pixel_data.groupby(['cell_index']).mean()
        cell_sum = pixel_data.groupby(['cell_index']).sum()
        cell_max = pixel_data.groupby(['cell_index']).max()

        vesicle_mean['vesicle_size'] = vesicle_sum.vesicle_mask

        cell_sum_dict = cell_sum.to_dict()
        cell_mean_dict = cell_mean.to_dict()
        cell_max_dict = cell_max.to_dict()

        vesicle_mean['cell_size'] = vesicle_mean.cell_index.apply(lambda x: cell_sum_dict['cell_mask'][x])
        vesicle_mean['cell_arrestin_mean'] = vesicle_mean.cell_index.apply(lambda x: cell_mean_dict['arrestin_fluo'][x])
        vesicle_mean['cell_L10_mean'] = vesicle_mean.cell_index.apply(lambda x: cell_mean_dict['L10_fluo'][x])
        vesicle_mean['cell_arrestin_max'] = vesicle_mean.cell_index.apply(lambda x: cell_max_dict['arrestin_fluo'][x])
        
        # drop the non-cell pixels
        vesicle_mean = vesicle_mean[vesicle_mean.cell_index != 0]
        
        return vesicle_mean

    def analyze_experiment(self):
        results = []
        czis = os.listdir(self.folder)
        czis = [czi for czi in czis if '.czi' in czi]
        for czi in tqdm(czis):
            name = czi.split(".czi")[0]
            
            for T in tqdm(range(self.no_timepoints)):

                img = ZeissCziImage(self.folder, czi)
                img.load_image(T=T)

                arrestin = img.image[:,:,self.vesicle_channel, 0]
                cells = img.image[:,:,self.cell_channel, 0]

                cell_idxs = io.imread(self.mask_folder / Path(f"cells_T{T}_{name}.tif"))
                vesicle_idxs = io.imread(self.mask_folder / Path(f"vesicles_T{T}_{name}.tif"))

                vesicle_masks = (vesicle_idxs > 0).astype(int)
                cell_masks = (cell_idxs > 0).astype(int)

                data = self.analyze_image(arrestin, cells, cell_idxs, vesicle_idxs, cell_masks, vesicle_masks)
                data["setup"] = name
                data["time"] = T
                data["setup_time"] = f"{name}_T_{T}"

                results.append(data)

        return pd.concat(results, ignore_index=True)


class ImageXpressExperiment(Experiment):
    
    def __init__(self, detector, folder, channel_names = {1:'1', 2:'2', 3:'3'}, channel_to_analyse = 2, calculate_background=True, no_bkgrd_imgs = 100):
        super().__init__(detector, folder, channel_names, channel_to_analyse)
        self.no_bkgrd_imgs = no_bkgrd_imgs
        self.get_images()
        if calculate_background:
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
        imgs = np.random.choice(list(self.images.keys()), self.no_bkgrd_imgs)
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
        
    def save_background(self, path):
        """Saves the background

        Args:
            path (str): path to save the background
        """
        # saves the self.background
        if type(self.background) != int:
            io.imsave(path, self.background)
        else:
            print("No background calculated yet")
            
    def load_background(self, path):
        """Loads the background

        Args:
            path (str): path of the background image
        """
        self.background = io.imread(path)
        
    def display_cells(self, subset, cells_in_row, channels, size=300, max_intensity=65535, seed=1, gamma=1, imshow_kwds={}, grayscale=True):
        """Displays the selected cells

        Displays a collection of cells by IDs in a single plot. Useful for checking a collection a cells with same properties (e.g. size, fluorescence, etc.)
        Args:
            subset (DataFrame): the subset of cells from which random cells are selected
            cells_in_row (int): the number of the displayed cells in a single row
            channels (list): the channels to display            
            size (int): the size of the image of single cells
            max_intensity (int): the max intensity on which the image will be normalized (devided by)
            seed (int): random seed for the selection
            gamma (float): the gamma to be applied for the image displayed
        Returns:
            Plot with the displayed cells
        """
        
        #this line should be removed later
        self.results.cell_id = range(len(self.results))
        
                
        channels_not_shown = [x for x in range(3) if x not in channels]    
        
        
        # remove background data if it exists
        subset = subset[(subset.cell_id > 0)]

        no_of_cells = cells_in_row * cells_in_row 
        
        if no_of_cells > len(subset):
            cells_in_row = int(np.sqrt(len(subset)))
            no_of_cells = cells_in_row * cells_in_row 
            
        concatenated = np.zeros((cells_in_row*size, cells_in_row*size, 3))
            
        np.random.seed(seed)
        cellIds = list(np.random.choice(subset.cell_id.values, no_of_cells, replace=False))
                
        print(cellIds)
        idx = 0
        for x in range(cells_in_row):
            for y in range(cells_in_row):
                
                Id = cellIds[idx]
                cell = self.select_cell(Id, size)
                x_fill = size - cell.shape[1]
                y_fill = size - cell.shape[0]
                cell = np.pad(cell, ((0,y_fill),(0,x_fill), (0,0)), 'constant', constant_values=(0))               
                
                concatenated[x*size:x*size+size, y*size:y*size+size, :] = cell
                idx +=1
               
               
        for channel in channels_not_shown:
            concatenated[:,:, channel] = 0       
        concatenated=concatenated/max_intensity    
        concatenated=np.clip(concatenated, 0, 1)
        
        #adjust gamma
        concatenated = exposure.adjust_gamma(concatenated, gamma)  
        
        if grayscale:
            concatenated = concatenated.sum(axis=2)
                    
        plt.figure(figsize=(cells_in_row*2,cells_in_row*2))     
        plt.imshow(concatenated, **imshow_kwds)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.show()

class Analyzer:
    
    def __init__(self, detector, channel_to_analyse):
        
        self.detector = detector
    
    def analyze(self, img, background, channel_to_analyse):
        # remove background first
        # img = img.astype(int) - background
        # img[img < 0] = 0
        img = img.astype(np.uint16)

        nuclei = self.detector.predict_nuclei(img)
        cells = self.detector.predict_cells(img)
        
        
        # remove background
        img = img - background
        img[img < 0] = 0
        img = img.astype(np.uint16)

        # add coordinates to pixels
        xv, yv = np.meshgrid(range(cells.shape[1]), range(cells.shape[0]))
        xv = xv.reshape(-1,1)
        yv = yv.reshape(-1,1)
        
        
        img = img[:,:,channel_to_analyse-1].reshape(-1,1)
        nuclei = (nuclei.reshape(-1,1) > 0).astype(int) # convert the mask ids to ones
        cells = cells.reshape(-1,1)
        background = background[:,:,channel_to_analyse-1].reshape(-1,1)
        cell_fluorescence = (cells>0) * img 
        nucleus_fluorescence = nuclei * img

     
        data = np.concatenate((cells, img, nuclei, nucleus_fluorescence, xv, yv, background), axis = 1)
        
        data = pd.DataFrame(data, columns = ['cell_id', 'cell_fluorescence', 'nucleus_size', 'nucleus_fluorescence', 'x_coor', 'y_coor', 'background'])
        
        data['cell_size'] = 1
        
        coords = data.groupby('cell_id').mean().reset_index()
        data = data.groupby('cell_id').sum().reset_index()
        
        data.x_coor = coords.x_coor.astype(int)
        data.y_coor = coords.y_coor.astype(int)

        data = data[data.cell_id > 0] # drop the background
        
        data["cytoplasm_fluorescence"] = data.cell_fluorescence - data.nucleus_fluorescence
        data["mean_cytoplasm_fluorescence"] = data.cytoplasm_fluorescence / (data.cell_size - data.nucleus_size)
        data["mean_nucleus_fluorescence"] = data.nucleus_fluorescence / data.nucleus_size
        data["mean_background_fluorescence"] = data.background / data.cell_size
        data["ratio"] = data.cytoplasm_fluorescence / data.nucleus_fluorescence
        data["mean_ratio"] = data.mean_cytoplasm_fluorescence / data.mean_nucleus_fluorescence

        return data
