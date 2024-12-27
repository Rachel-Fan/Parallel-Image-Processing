import os
import time
import numpy as np
import cv2
import pandas as pd
from numba import cuda
from src.kernels.edge_detection import edge_detection, d_sobel_x, d_sobel_y
from src.kernels.histogram_equalization import histogram_equalization

# CPU implementations of the kernels
def cpu_histogram_equalization(input_image):
    hist = np.zeros(256, dtype=np.int32)
    for value in input_image.ravel():
        hist[value] += 1

    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]  # First non-zero value of CDF
    total_pixels = input_image.size

    output_image = np.zeros_like(input_image, dtype=np.uint8)
    for x in range(input_image.shape[0]):
        for y in range(input_image.shape[1]):
            pixel_value = input_image[x, y]
            output_image[x, y] = int((hist[pixel_value] - cdf_min) / (total_pixels - cdf_min) * 255)

    return output_image

def cpu_edge_detection(input_image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    output_image = np.zeros_like(input_image, dtype=np.uint8)

    for x in range(1, input_image.shape[0] - 1):
        for y in range(1, input_image.shape[1] - 1):
            gx = 0.0
            gy = 0.0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    gx += input_image[x + i, y + j] * sobel_x[i + 1, j + 1]
                    gy += input_image[x + i, y + j] * sobel_y[i + 1, j + 1]

            output_image[x, y] = min(255, int(np.sqrt(gx**2 + gy**2)))

    return output_image

# GPU test function
def gpu_process(input_image):
    output_image_hist_eq = np.zeros_like(input_image, dtype=np.uint8)
    output_image_edge = np.zeros_like(input_image, dtype=np.uint8)

    # Histogram Equalization
    hist = np.zeros(256, dtype=np.int32)
    for value in input_image.ravel():
        hist[value] += 1

    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    total_pixels = input_image.size

    d_input_image = cuda.to_device(input_image)
    d_output_hist_eq = cuda.to_device(output_image_hist_eq)
    d_hist = cuda.to_device(hist)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (input_image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (input_image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    histogram_equalization[blocks_per_grid, threads_per_block](
        d_input_image, d_output_hist_eq, d_hist, cdf_min, total_pixels
    )
    output_image_hist_eq = d_output_hist_eq.copy_to_host()

    # Edge Detection
    d_output_edge = cuda.to_device(output_image_edge)
    edge_detection[blocks_per_grid, threads_per_block](
        d_input_image, d_output_edge, d_sobel_x, d_sobel_y
    )
    output_image_edge = d_output_edge.copy_to_host()

    return output_image_hist_eq, output_image_edge

def main():
    input_image_path = os.path.join('data', 'input_image.jpg')
    output_folder = os.path.join('data', 'output')
    os.makedirs(output_folder, exist_ok=True)

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    # CPU performance
    start_cpu_he = time.time()
    cpu_hist_eq = cpu_histogram_equalization(input_image)
    end_cpu_he = time.time()

    start_cpu_edge = time.time()
    cpu_edge = cpu_edge_detection(input_image)
    end_cpu_edge = time.time()

    # GPU performance
    start_gpu_he = time.time()
    gpu_hist_eq, _ = gpu_process(input_image)
    end_gpu_he = time.time()

    start_gpu_edge = time.time()
    _, gpu_edge = gpu_process(input_image)
    end_gpu_edge = time.time()

    # Performance results
    cpu_he_time = end_cpu_he - start_cpu_he
    cpu_edge_time = end_cpu_edge - start_cpu_edge
    gpu_he_time = end_gpu_he - start_gpu_he
    gpu_edge_time = end_gpu_edge - start_gpu_edge

    print("CPU Histogram Equalization Time:", cpu_he_time, "seconds")
    print("CPU Edge Detection Time:", cpu_edge_time, "seconds")
    print("GPU Histogram Equalization Time:", gpu_he_time, "seconds")
    print("GPU Edge Detection Time:", gpu_edge_time, "seconds")

    # Save results to CSV
    results = {
        "Kernel": ["Histogram Equalization", "Edge Detection"],
        "CPU Time (s)": [cpu_he_time, cpu_edge_time],
        "GPU Time (s)": [gpu_he_time, gpu_edge_time]
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "performance_comparison_detailed.csv"), index=False)

    # Save output images
    cv2.imwrite(os.path.join(output_folder, "cpu_hist_eq.jpg"), cpu_hist_eq)
    cv2.imwrite(os.path.join(output_folder, "cpu_edge.jpg"), cpu_edge)
    cv2.imwrite(os.path.join(output_folder, "gpu_hist_eq.jpg"), gpu_hist_eq)
    cv2.imwrite(os.path.join(output_folder, "gpu_edge.jpg"), gpu_edge)

    # Display results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_image, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("CPU Histogram Equalization")
    plt.imshow(cpu_hist_eq, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("CPU Edge Detection")
    plt.imshow(cpu_edge, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("GPU Histogram Equalization")
    plt.imshow(gpu_hist_eq, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("GPU Edge Detection")
    plt.imshow(gpu_edge, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "CPU_GPU_comparison.png"))
    plt.show()

if __name__ == "__main__":
    main()
