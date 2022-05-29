import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from io import BytesIO
from cellanalysis import cellanalysis as ca

st.set_page_config(layout="wide")

st.title("Image analysis app")


with st.sidebar:
    if st.button('Say hello'):
        st.write('Why hello there')
    else:
        st.write('Goodbye')

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:

    # define column layout
    col1, col2, col3 = st.columns(3)

    img = io.imread(uploaded_file)

    img = (img / (img.max() / 255)).astype(int)

    with col1:
        st.image(img, width=300)

    ### image analysis
    detector = ca.CellDetector()
    img = img.reshape(img.shape[0], img.shape[1], 1)
    masks = detector.predict_nuclei(img, nucleus_channel=1)
    masks = masks / masks.max()

    # display segmented
    with col2:
        st.image(masks, width=300)

    buf = BytesIO()
    io.imsave(buf, masks, format='png')
    byte_im = buf.getvalue()

    with col3:
        st.download_button("Download nuclei mask", data=byte_im, file_name="segmented.png",  mime="image/png")

