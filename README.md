# GPU-Accelerated Image Processing Project

## Purpose of the Project

This project demonstrates the implementation and performance comparison of GPU-accelerated image processing kernels, focusing on Edge Detection and Histogram Equalization. The primary goal is to leverage GPU parallelism for computationally intensive image processing tasks and to visualize the results.

By utilizing CUDA and Python (Numba library), this project aims to:

1. Highlight the speedup achieved with GPU acceleration.
2. Provide a foundation for applying GPU-based optimization techniques in real-world image processing pipelines.
3. Enable intuitive visualization of processing results.

## Mathematical Explanation of Kernels

### 1. Edge Detection (Sobel Operator)

The Sobel Operator computes the gradient magnitude of an image to highlight edges. Two kernels are used:

1. **Sobel X Kernel:**  
   Detects horizontal edges.

2. **Sobel Y Kernel:**  
   Detects vertical edges.

The gradient magnitude is computed as:
\[
G(x, y) = \sqrt{(G_x(x, y))^2 + (G_y(x, y))^2}
\]
Where:
- \(G_x(x, y)\) is the gradient in the x-direction.
- \(G_y(x, y)\) is the gradient in the y-direction.
- \(G(x, y)\) represents the edge strength at pixel \((x, y)\).

### 2. Histogram Equalization

Histogram Equalization enhances the contrast of an image by redistributing its pixel intensity values, ensuring uniform brightness across the image.

## Features and Outputs

### `main.py`

1. **Input**: `input_image.jpg` from the `data/` folder.
2. **Processing**:
   - Edge Detection on the original image.
   - Histogram Equalization on the original image.
   - Edge Detection on the equalized image.
3. **Output**:
   - A plot showing the original image and the three processed images will be displayed in a new window.
   - The plot is saved as `data/output/GPU_image_processing.png`.

### `performance_comparison.py`

1. **Input**: `input_image.jpg` from the `data/` folder.
2. **Processing**:
   - CPU Edge Detection.
   - CPU Histogram Equalization.
   - GPU Edge Detection.
   - GPU Histogram Equalization.
3. **Output**:
   - A comparison plot (`CPU_GPU_comparison.png`) showing the original image and the four processed images is saved in `data/output/`.
   - Performance metrics (processing time for each method) are recorded, printed, and saved in `data/output/performance_comparison_detailed.csv`.

### Notebooks

#### `image_processing.ipynb`
1. **Input**: Images from `Notebook/data/input_images`.
2. **Processing**:
   - Edge Detection and Histogram Equalization.
3. **Output**:
   - The input image dimensions and processing time for each method are printed.
   - Results are visualized in the notebook.
   - Processed images are saved as `edge_{input_image_name}.jpg` and `hist_eq_{input_image_name}.jpg` in `Notebook/data/output_images`.

#### `video_processing.ipynb`
1. **Input**: A 10-second `input_video.mp4` from `Notebook/data/`.
2. **Processing**:
   - Edge Detection applied to the video frame-by-frame.
3. **Output**:
   - Frames will be processed and saved in `Notebook/data/output_frames`.
   - The processed video is saved as `edge_detection_video.avi` in `Notebook/data/`.
   - A GIF (`edge_detection.gif`) is generated for visualization and saved in `Notebook/data/`.

## Performance Benchmarks

The GPU acceleration significantly outperforms traditional CPU methods, particularly for computationally intensive edge detection tasks. GPU processing demonstrates a notable reduction in processing times by leveraging parallelism.

## Instructions for Running the Code

### Prerequisites

1. Install Python (3.8 or higher) and the necessary dependencies:
   ```bash
   pip install numba numpy opencv-python matplotlib tqdm
2. Ensure a CUDA-compatible GPU is available and CUDA drivers are installed.

## Running the Code

### For Image Processing

1. Place an input image in the data/ folder and name it input_image.jpg
2. Run the script:
   ```bash
   python main.py

### For Performance Comparison

1. Place an input image in the data/ folder and name it input_image.jpg
2. Run the script:
   ```bash
   python performance_comparison.py

### Notebooks

-  Open `image_processing.ipynb` to process and visualize images.
-  Open `video_processing.ipynb` to process and visualize images.

## Output

-  Processed Images: Saved `Notebook/data/output_images/`.
-  Processed Video: (`edge_detection_video.avi`) in `Notebook/data/`.
-  GIF for Visualization: (`edge_detection.gif`) in `Notebook/data/`.


