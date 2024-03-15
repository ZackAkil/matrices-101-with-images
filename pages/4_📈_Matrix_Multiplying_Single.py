import streamlit as st
import numpy as np

import matplotlib.pyplot as plt




st.set_page_config(layout="wide")
st.title("Matrix multiplier - single number")


multiply = st.slider('Multiply by:', 0, 10, 1)


cmap_option = st.selectbox(
"Colormap:", 
options=['Greys', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'jet', 'cool']
) 

viz = st.checkbox('Vizualise')

col1, col2 = st.columns(2)

def diplay_matrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix,  vmin=0, vmax=10, cmap=cmap_option)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


with col1:

    # Create the matrix when a button is clicked
    # if st.button("Create Matrix"):
    np.random.seed(42)
    matrix = np.random.randint(1, 10, size=(5, 5))
    st.write("original matrix")
    st.dataframe(matrix)  # Simple visualization using a DataFrame
    if viz:
        diplay_matrix(matrix)


with col2:
    st.write("original matrix multiplied by ", multiply )
    # Get user input for dimensions
    matrix_multiply = matrix * multiply
    st.dataframe(matrix_multiply)  # Simple visualization using a DataFrame

    if viz:
        diplay_matrix(matrix_multiply)