import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to resize the image
def resize_image(image, size):
    return image.resize(size)

def display_matrix(matrix):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,  vmin=0, vmax=10, 
                #    cmap='cool'
                   )
    # fig.colorbar(im, ax=ax)
    st.pyplot(fig)



# Title and description
st.title("Image Editor")



size_options = st.selectbox(
"Sizes:", 
options=[10,50,100]
)  

col1, col2, col3 = st.columns(3)

with col1:
    # File uploader component
    uploaded_file_1 = st.file_uploader("Image 2", key='1', type=["jpg", "jpeg"])
    viz = st.checkbox('Camera')
    if viz:
        uploaded_file_1 = st.camera_input("Take a picture")

    multiply = st.slider('Multiply by:', 0., 3., 1., .2, key='alkasajhsak')


    # Process and display the image
    if uploaded_file_1 is not None:
        image_1 = Image.open(uploaded_file_1)
        resized_image_1 = resize_image(image_1, (size_options, size_options))
        numpy_array_1 =( np.array(resized_image_1).astype(float)  * multiply).astype(int)
        st.write(numpy_array_1.shape)
        display_matrix(numpy_array_1)

with col2:
    # File uploader component
    uploaded_file_2 = st.file_uploader("Image 2", key='2', type=["jpg", "jpeg"])
    multiply_2 = st.slider('Multiply by:', 0., 3., 1., .2, key='alkjhsak')

    # Process and display the image
    if uploaded_file_2 is not None:
        image_2 = Image.open(uploaded_file_2)
        resized_image_2 = resize_image(image_2, (size_options, size_options))
        numpy_array_2 =( np.array(resized_image_2).astype(float)  * multiply_2).astype(int)
        st.write(numpy_array_2.shape)
        display_matrix(numpy_array_2)


def multiply_matrix(m1, m2):

    return ((((m1.astype(float))/255.)*((m2.astype(float))/255.))*255.).astype(int)

with col3:

    if st.button('Add'):
        try:
            # st.write(numpy_array_1 + numpy_array_2)

            matrix_sum =  numpy_array_1 + numpy_array_2
            display_matrix(np.clip(matrix_sum, 0, 255))
        except Exception as e:
            st.warning(e, icon="⚠️")

    if st.button('Subtract'):
        try:
            # st.write(numpy_array_1 + numpy_array_2)
            matrix_sum = numpy_array_1 - numpy_array_2
            display_matrix(np.clip(matrix_sum, 0, 255))
        except Exception as e:
            st.warning(e, icon="⚠️")

    if st.button('Multiply'):
        try:
            # st.write(numpy_array_1 + numpy_array_2)
            matrix_sum = multiply_matrix(numpy_array_1, numpy_array_2)
            display_matrix(np.clip(matrix_sum, 0, 255))
        except Exception as e:
            st.warning(e, icon="⚠️")




