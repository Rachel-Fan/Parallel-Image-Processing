# GPU-Accelerated Image Processing Project

## Purpose of the Project

This project demonstrates the implementation and performance comparison of GPU-accelerated image processing kernels with their CPU counterparts. The primary goal is to leverage GPU parallelism for computationally intensive image processing tasks, such as Histogram Equalization and Edge Detection, and to evaluate their efficiency over traditional CPU-based methods.

By utilizing CUDA and Python (Numba library), this project aims to:

1. Highlight the speedup achieved with GPU acceleration.
2. Provide a foundation for applying GPU-based optimization techniques in real-world image processing pipelines.

## Mathematical Explanations of Each Kernel

### 1. Histogram Equalization

Histogram Equalization enhances the contrast of an image by redistributing its pixel intensity values.

Mathematical Formula:

Let:

H(v) be the histogram value for pixel intensity v.

CDF(v) be the cumulative distribution function up to intensity v.

CDF_min be the minimum non-zero CDF value.

N be the total number of pixels in the image.

The new pixel intensity is calculated as:

Where:

P(x, y) is the original pixel value at position (x, y).

P'(x, y) is the enhanced pixel value.

### 2. Edge Detection (Sobel Operator)

The Sobel Operator computes the gradient magnitude of an image to highlight edges. Two kernels are used:

Sobel X Kernel:

Sobel Y Kernel:

The gradient magnitude is computed as:

Where:

G(x, y) is the gradient magnitude at pixel (x, y).

## Performance Benchmarks

The following results highlight the processing times for both CPU and GPU implementations:

### Observations

GPU acceleration achieved significant speedups for both kernels, especially for computationally intensive Edge Detection.

Histogram Equalization benefited moderately from GPU optimization due to its lower computational complexity compared to Edge Detection.

### Optimizations

Parallelism: GPU kernels leverage thousands of threads to process multiple pixels simultaneously.

Memory Access: Input and output images are allocated on the GPU memory, reducing data transfer overhead.

Grid and Block Configuration: The kernel launches are optimized for 16x16 thread blocks, balancing thread utilization and memory efficiency.

## Instructions for Running the Code

### Prerequisites

Install Python (3.8 or higher) and ensure you have the necessary dependencies:

pip install numba numpy opencv-python matplotlib pandas

Ensure a CUDA-compatible GPU is available and CUDA drivers are installed.

Running the Code

Clone the repository:

git clone <repository-url>
cd <repository-folder>

Place an input image in the data folder and name it input_image.jpg.

Execute the script:

python main.py

Output

Processed images and the comparison plot (CPU_GPU_comparison.png) will be saved in the data folder.

Performance metrics will be saved in a CSV file (performance_comparison_detailed.csv) in the same folder.

Future Work

Extend the implementation to handle color images.

Add additional GPU kernels, such as Gaussian Blur and Median Filtering.

Optimize memory usage further by leveraging shared memory in CUDA.

Implement real-time video processing pipelines using similar kernels.
