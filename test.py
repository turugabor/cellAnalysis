# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from cellanalysis import cellanalysis as ca

# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_closing
from skimage import measure
import pandas as pd

# %%
import importlib

# %%
importlib.reload(ca)

# %%
imgZ = ca.ZeissCziImage(folder = "data/ZeissConfocalSamples", name = "3_20x.czi", channel_names={1:"cells"})

# %%
imgZ.load_image()

# %%
imgZ.image.shape

# %%
imgZ.display_image()

# %%
imgX = ca.ImageXpressImage(folder = "data/imageXpressSamples", name = "A01_s10", channel_names={1:"DAPI", 2:'egyik', 3:'masik'})

# %%
imgX.load_image()

# %%
imgX.image.shape

# %%
imgX.display_image()

# %%
imgX.subtract_background()

# %%
<<<<<<< HEAD
nuclei = detector.predict_nuclei(imgX.image)
plt.imshow(nuclei)

# %%
=======
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
detector = ca.CellDetector()

# %%
analyzer = ca.Analyzer(detector)

# %%
data = analyzer.analyze(imgX.image, 1)

# %%
data

# %%
cells = detector.predict_cells(imgX.image, cell_channel=3)
plt.imshow(cells)

# %%
nucleus_img = imgX.image[:, :, 2]


# %%
plt.imshow(nucleus_img)

# %%
filtered = (nuclei > 0) * nucleus_img

# %%
plt.imshow(filtered)

# %%
filtered.sum() / (nuclei > 0).sum()

# %%
plt.imshow(nuclei == 3)

# %%
(nuclei == 3).sum()

# %%
# idx | nucleus_size | nucleus_fluoreescence

# %%
nuclei.reshape(-1,1) 

# %%
nucleus_img.reshape(-1,1)

# %%
joint = np.concatenate([nuclei.reshape(-1,1) , nucleus_img.reshape(-1,1)], axis=1)

# %%
joint.shape

# %%
data = pd.DataFrame(joint, columns=['cell_idx', 'fluorescence'])

# %%
data = data[data.cell_idx > 0]

# %%
analizer = ca.Analyzer()

# %%
analized = analizer.analyze(imgX.image,nuclei)

# %%
analized
