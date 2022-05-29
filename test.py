# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
<<<<<<< HEAD
#       format_name: light
#       format_version: '1.5'
=======
#       format_name: percent
#       format_version: '1.3'
<<<<<<< HEAD
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
#       jupytext_version: 1.13.7
=======
#       jupytext_version: 1.13.8
>>>>>>> 2f3f4b7cdd2e08073fcce05903482cb56e06b96a
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from cellanalysis import cellanalysis as ca
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
from cellpose import models
>>>>>>> 2f3f4b7cdd2e08073fcce05903482cb56e06b96a

# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_closing
from skimage import measure
import pandas as pd
import re

# %%
import importlib

# %%
importlib.reload(ca)

# %%
exp = ca.ImageXpressExperiment(folder = "data/imageXpressSamples/")

# %%
exp.get_images()

# %%
exp.images

# %%
exp.images['B01_s13'].load_image()

# %%
exp.images['B01_s13'].display_image()

# %%
detector = ca.CellPoseDetector()

# %%
detector = ca.CellDetector()

# %%
img = ca.ImageXpressImage('data/imageXpressSamples/', 'A01_s15' )

img.load_image()

masks = detector.predict_nuclei(img.image[:,:,0].reshape(img.image.shape[0], img.image.shape[1], 1), nucleus_channel=1)

# %%
img.image.shape

# %%
plt.imshow(masks)

# %%

# %%

# %%

# %%
exp.analyse_experiment()

# %%

# %%

# %%
imgX = ca.ImageXpressImage(folder = "data/20220504", name = "A01_s1", channel_names={1:"DAPI", 2:'egyik', 3:'masik'})

# %%
imgX.load_image()

# %%
imgX.subtract_background()

# %%
imgX.image.shape

# %%
imgX.display_image()

# %%
detector = ca.CellPoseDetector()

# %%
nuclei = detector.predict_nuclei(imgX.image)
plt.imshow(nuclei)

# %%
cells = detector.predict_cells(imgX.image[:512,:512], channels=[3,1], do_3D=False)
plt.imshow(cells)

# %%
analyzer = ca.Analyzer(detector)

# %%
data = analyzer.analyze(imgX.image, 1)

# %%
data

# %%
imgX.image.shape

# %%
cell_model = models.Cellpose(gpu=False, model_type='cyto')

# %%
channels = [3,1]
masks, flows, styles, diams = cell_model.eval(imgX.image, channels=channels, diameter=300)

# %%
masks, flows, styles, diams = cell_model.eval(imgX.image, channels=channels, diameter=300, )

# %%
masks.shape

# %%
plt.imshow(masks)
plt.show()

# %%
plt.imshow(cells)

# %%
imgX.display_image()

# %%
<<<<<<< HEAD
analized
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
=======
>>>>>>> 2f3f4b7cdd2e08073fcce05903482cb56e06b96a
