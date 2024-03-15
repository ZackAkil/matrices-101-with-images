import streamlit as st
import numpy as np

import matplotlib.pyplot as plt



st.set_page_config(layout="wide")
st.title("Matrix Subtracting")

col1, col2, col3 = st.columns(3)


def display_matrix(matrix):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,  vmin=0, vmax=10, cmap='cool')
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
    st.header("Matrix 1 - Matrix 2")
    if st.button('calculate'):
        try:
            st.write(matrix- matrix2)
            display_matrix(matrix- matrix2)
        except Exception as e:
            st.warning(e, icon="⚠️")

