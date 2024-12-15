import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly import tensor
import time
import cv2
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2

st.set_page_config(
    page_title="Skripsi",
    layout='centered'
)

# Preprocess
def preprocess(image):
    image_resized = transform.resize(image, (512, 512))
    return 0.299 * image_resized[:, :, 0] + 0.587 * image_resized[:, :, 1] + 0.114 * image_resized[:, :, 2]

def preprocessing(image):
            image_resized = cv2.resize(image, (512,512))
            return 0.299 * image_resized[:, :, 0] + 0.587 * image_resized[:, :, 1] + 0.114 * image_resized[:, :, 2]
        
# Tensor Decomposition
def tensor_decomposition(image, rank):
    factors = parafac(tensor(image), rank)
    reconstructed_tensor = tl.kruskal_to_tensor(factors)
    return reconstructed_tensor

# PSNR calculation
def PSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    max_pixel_value = 255
    psnr = 10 * np.log10((max_pixel_value)**2 / mse)
    return psnr

# Streamlit app
st.title("Denoising")
st.sidebar.subheader("Parameters")

# File upload
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
uploaded_noise_file = st.file_uploader("Upload a noisy image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file and uploaded_noise_file:
    image = io.imread(uploaded_file)
    image_noise = io.imread(uploaded_noise_file)

    # Select algorithm
    algorithm = st.selectbox("Select Denoising Algorithm", ["Deconvolution", "Non-Local Means", "Tensor Decomposition"])
    
    if algorithm == "Deconvolution":
        
        image_gray = preprocess(image)
        image_gray_noise = preprocess(image_noise)
        
        sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 2.0)
        kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        
        def gaussian_kernel(size, sigma):
            ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / (2 * sigma ** 2))
            return kernel / np.sum(kernel)
        
        K = st.sidebar.slider("K (Denoise Power)", 0.1, 50.0, 20.0)
        
        def wiener_deconvolution(img, kernel, K):
            kernel /= np.sum(kernel)
            img_fft = fft2(img)
            kernel_fft = fft2(kernel, img.shape)
            kernel_fft_conj = np.conj(kernel_fft)
            denominator = np.abs(kernel_fft) ** 2 + K
            wiener_filter = kernel_fft_conj / denominator
            deconvolved_fft = img_fft * wiener_filter
            deconvolved_image = np.abs(ifft2(deconvolved_fft))
            return deconvolved_image

        kernel = gaussian_kernel(kernel_size, sigma)
        start_time = time.time()
        convolutions = convolve2d(image_gray_noise, kernel, mode='same', boundary='wrap')
        denoised_image = wiener_deconvolution(convolutions, kernel, K)
        runtime = time.time() - start_time

    elif algorithm == "Non-Local Means":
        image_gray = preprocessing(image).astype(np.uint8)
        image_gray_noise = preprocessing(image_noise).astype(np.uint8)
        
        h = st.sidebar.slider("Filter Strength (h)", 1, 30, 10)
        patch_size = st.sidebar.slider("Patch Size", 1, 15, 7)
        search_window_size = st.sidebar.slider("Search Window Size", 10, 50, 21)
        
        def non_local_means(image, patch_size, search_window_size, h):
            start_time = time.time()
            denoised = cv2.fastNlMeansDenoising(image, h, patch_size, search_window_size)
            end_time = time.time()
            print(f"runtime of denoising: {end_time - start_time: .5f} seconds")
            return denoised

        start_time = time.time()
        denoised_image = non_local_means(image_gray_noise, h, patch_size, search_window_size)
        runtime = time.time() - start_time

    elif algorithm == "Tensor Decomposition":
        image_gray = preprocess(image)
        image_gray_noise = preprocess(image_noise)
        
        rank = st.sidebar.slider("Tensor Rank", 10, 500, 100)

        def tensor_decomposition(image, rank):
            factors = parafac(image, rank)
            reconstucted_tensor = tl.kruskal_to_tensor(factors)
            return reconstucted_tensor
        
        start_time = time.time()
        tensor_image_noise = tensor(image_gray_noise)
        denoised_image = tensor_decomposition(tensor_image_noise, rank)
        runtime = time.time() - start_time

    # PSNR Calculation
    psnr_value = PSNR(image_gray, denoised_image)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Grayscale Image")
    axes[0].axis("off")

    axes[1].imshow(image_gray_noise, cmap="gray")
    axes[1].set_title("Noisy Grayscale Image")
    axes[1].axis("off")

    axes[2].imshow(denoised_image, cmap="gray")
    axes[2].set_title("Denoised Image")
    axes[2].axis("off")

    st.pyplot(fig)
    
    st.write(f"Runtime: {runtime:.5f} seconds")
    st.write(f"PSNR: {psnr_value:.5f} dB")