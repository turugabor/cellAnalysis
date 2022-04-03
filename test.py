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
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from cellanalysis import cellanalysis as ca
<<<<<<< HEAD
=======

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
imgZ = ca.ZeissCziImage("data/ZeissConfocalSamples/3_20x.czi")

# %%
imgX = ca.ImageXpressImage("data/imageXpressSamples", "A01", "s10")

# %%
imgX.load_image()

# %%
imgX.image.shape

# %%
imgX.display_image()

# %%
imgX.subtract_background()

# %%
detector = ca.CellDetector()

# %%
nuclei = detector.predict_nuclei(imgX.image)

# %%
plt.imshow(nuclei)

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
>>>>>>> c927c50bfad8d4d2270e77dc3e9d69b1c8fccb85
