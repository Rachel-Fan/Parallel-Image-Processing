import os
import math
from numba import cuda
import numpy as np
import cv2  # Add OpenCV for image reading
from src.kernels.edge_detection import edge_detection, d_sobel_x, d_sobel_y
from src.kernels.histogram_equalization import histogram_equalization

def process_images(input_image):
    output_image_edge_original = np.zeros_like(input_image)
    output_image_edge_hist_eq = np.zeros_like(input_image)
    output_image_hist_eq = np.zeros_like(input_image, dtype=np.uint8)

    # Calculate histogram on host
    hist = np.zeros(256, dtype=np.int32)
    for value in input_image.ravel():
        hist[value] += 1

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]  # First non-zero value of CDF
    total_pixels = input_image.size

    # Allocate memory on device
    d_input_image = cuda.to_device(input_image)
    d_output_edge_original = cuda.to_device(output_image_edge_original)
    d_output_edge_hist_eq = cuda.to_device(output_image_edge_hist_eq)
    d_output_hist_eq = cuda.to_device(output_image_hist_eq)
    d_hist = cuda.to_device(hist)

    # Define block and grid sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(input_image.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(input_image.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch Histogram Equalization kernel
    histogram_equalization[blocks_per_grid, threads_per_block](
        d_input_image, d_output_hist_eq, d_hist, cdf_min, total_pixels
    )

    # Copy histogram equalized result back to host
    output_image_hist_eq = d_output_hist_eq.copy_to_host()

    # Launch Edge Detection kernel on the original image
    edge_detection[blocks_per_grid, threads_per_block](
        d_input_image, d_output_edge_original, d_sobel_x, d_sobel_y
    )

    # Launch Edge Detection kernel on histogram equalized image
    d_hist_eq_input_image = cuda.to_device(output_image_hist_eq)
    edge_detection[blocks_per_grid, threads_per_block](
        d_hist_eq_input_image, d_output_edge_hist_eq, d_sobel_x, d_sobel_y
    )

    # Copy edge detection results back to host
    output_image_edge_original = d_output_edge_original.copy_to_host()
    output_image_edge_hist_eq = d_output_edge_hist_eq.copy_to_host()

    return output_image_hist_eq, output_image_edge_original, output_image_edge_hist_eq
