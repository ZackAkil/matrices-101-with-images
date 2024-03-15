import streamlit as st
import numpy as np

import matplotlib.pyplot as plt



st.set_page_config(layout="wide")
st.title("Matrix Multiplying")

cmap_option = st.selectbox(
"Colormap:", 
options=['Greys', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'jet', 'cool']
) 

colour_scale = st.slider('Colour Scale', 10, 100, 10)


col1, col2, col3 = st.columns(3)


def display_matrix(matrix):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,  vmin=0, vmax=colour_scale, cmap=cmap_option)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


with col1:
    st.header("Matrix 1")
    size1 = st.slider('Matrix 1 size', 1, 10, 1)
    np.random.seed(42)
    matrix = np.random.randint(1, 10, size=(size1, size1))
    st.write(matrix)

    display_matrix(matrix)

with col2:
    st.header("Matrix 2")

    size2 = st.slider('Matrix 2 size', 1, 10, 1)
    np.random.seed(43)
    matrix2 = np.random.randint(1, 10, size=(size2, size2))
    st.write(matrix2)
    display_matrix(matrix2)


with col3:
    st.header("Matrix 1 x Matrix 2")
    if st.button('calculate'):
        try:
            st.write(matrix * matrix2)
            display_matrix(matrix * matrix2)
        except Exception as e:
            st.warning(e, icon="⚠️")

