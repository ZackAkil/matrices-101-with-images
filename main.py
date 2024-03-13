import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


session_values = ['matrix1', 'matrix2', 'matrix_sum']

for value in session_values:
    if value not in st.session_state:
        st.session_state[value] = None

def generate_random_matrix(shape):
    """Generates a random matrix of the specified size."""
    return np.random.randint(1, 10, size=shape)

# def visualize_matrix(matrix, title="Matrix"):
#     """Visualizes a matrix using Matplotlib."""
#     st.write(matrix)
#     fig, ax = plt.subplots()
#     ax.imshow(matrix, cmap="hot")
#     # fig.colorbar(label="Matrix Values")
#     # fig.title(title)
#     # fig.xlabel("Columns")
#     # fig.ylabel("Rows")
#     fig.show()
#     st.pyplot(fig)

def visualize_matrix_image(matrix):
    fig, ax = plt.subplots()
    ax.imshow(matrix,cmap="cool" )
    fig.show()
    st.pyplot(fig)

def visualize_matrix(matrix, t):
    """
    This function visualizes a 2D matrix using a colormap and labels each cell with its value.

    Args:
        matrix: A 2D list or NumPy array representing the matrix.
        cmap (str, optional): The colormap to use for visualization. Defaults to "coolwarm".
    """
    st.write(matrix)
    fig, ax = plt.subplots()
    ax.imshow(matrix,cmap="cool" )

    # Add labels for each cell
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")

    # fig.colorbar(ax)  # Colorbar associated with the specific axis
    # ax.set_xlabel("Column Index")
    # ax.set_ylabel("Row Index")
    # ax.set_title("Matrix Visualization")
    fig.show()
    st.pyplot(fig)


def add_matrices(matrix1, matrix2):
    """Adds two matrices element-wise."""
    if matrix1.shape != matrix2.shape:
        st.error("Matrices must have the same dimensions for addition.")
        return None
    return matrix1 + matrix2

def apply_grayscale_filter(image):
    """Converts an image to grayscale (using NumPy)."""
    return np.mean(image, axis=2)  # Average across RGB channels for grayscale

def apply_brightness_filter(image, value):
    """Adjusts the brightness of an image using NumPy."""
    return np.clip(image + value, 0, 255)  # Clip values to 0-255 range

def main():
    """Main function of the Streamlit app."""

    # Title and introduction
    st.title("Interactive Matrix Playground")
    st.write("Learn about matrices and their applications in a fun and visual way!")

    # Matrix size selection
    matrix_size = st.number_input("Matrix Size", min_value=2, max_value=10, key="matrix_size")

    # Generate random matrices
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matrix 1")
        matrix1_button = st.button("Generate Random Matrix (Size {})".format(matrix_size), key="matrix1_button")
        
        if matrix1_button:
            matrix1 = generate_random_matrix((matrix_size, matrix_size))
            st.session_state.matrix1 = matrix1

        if st.session_state.matrix1 is not None:
            visualize_matrix(st.session_state.matrix1, "Matrix 1")

    with col2:
        st.subheader("Matrix 2")
        matrix2_button = st.button("Generate Random Matrix (Size {})".format(matrix_size), key="matrix2_button")
        
        if matrix2_button:
            matrix2 = generate_random_matrix((matrix_size, matrix_size))
            st.session_state.matrix2 = matrix2

        if st.session_state.matrix2 is not None:
            visualize_matrix(st.session_state.matrix2, "Matrix 2")
        

    # Add matrices button and visualization
    if st.button("Add Matrices", key="add_button"):
        st.session_state.matrix_sum = add_matrices(st.session_state.matrix1, st.session_state.matrix2)

    if st.session_state.matrix_sum is not None:
        visualize_matrix(st.session_state.matrix_sum, "Result of Matrix Addition")

    # Webcam image upload and filter application
    st.header("Image Filter with Matrices")
    st.write("Upload an image from your webcam and apply matrix-based filters!")

    # Use st.camera_input
    # image = st.camera_input("Webcam Image", key="webcam_image")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)

        # To convert PIL Image to numpy array:
        img_array = np.array(img)

        # Check the type of img_array:
        # Should output: <class 'numpy.ndarray'>
        # st.write(img_array)
        st.write(img_array.shape) 
        visualize_matrix_image(img_array)

        # Check the shape of img_array:
        # Should output shape: (height, width, channels)
        

    # if image is not None:
    #     # Convert to NumPy array for processing
    #     image = np.array(image)

    #     # Filter selection dropdown
    #     filter_options = ["Grayscale", "Brightness Adjustment"]
    #     selected_filter = st.selectbox("Apply Filter", filter_options, key="filter_select")

    #     if selected_filter == "Grayscale":
    #         filtered_image = apply_grayscale_filter(image.copy())
    #         st.write("Grayscale Image")
    #         st.image(filtered_image, channels="RGB", use_column_width=True)

    #     elif selected_filter == "Brightness Adjustment":
    #         brightness_value = st.slider("Brightness", min_value=-128, max_value=128, key="brightness_slider")
    #         filtered_image = apply_brightness_filter(image.copy(), brightness_value)
    #         st.write("Adjusted Brightness Image")
    #         st.image(filtered_image, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
