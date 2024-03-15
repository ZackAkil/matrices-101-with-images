import streamlit as st
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px
import numpy as np





st.set_page_config(layout="wide")
st.title("Matrix Creator")

col1, col2, col3 = st.columns(3)

with col1:
    # Get user input for dimensions
    rows = st.slider('Rows', 1, 10, 1)
    cols = st.slider('Columns', 1, 10, 1)

with col2:
    # Create the matrix when a button is clicked
    # if st.button("Create Matrix"):
    matrix = np.random.randint(1, 10, size=(rows, cols))
    st.write("Your Matrix shape:", matrix.shape)
    st.write("Your Matrix:")
    st.dataframe(matrix)  # Simple visualization using a DataFrame


with col3:
    viz = st.checkbox('Vizualise')
    if viz:
        cmap_option = st.selectbox(
        "Colormap:", 
        options=['Greys', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'jet', 'cool']
        )  


        fig, ax = plt.subplots()
        im = ax.imshow(matrix,  vmin=0, vmax=10, cmap=cmap_option)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        