import math
from numba import cuda
import numpy as np

# Sobel kernels defined as NumPy arrays
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.int32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.int32)

# Copy Sobel kernels to constant memory
d_sobel_x = cuda.to_device(sobel_x)
d_sobel_y = cuda.to_device(sobel_y)

@cuda.jit
def edge_detection(input_image, output_image, sobel_x, sobel_y):
    x, y = cuda.grid(2)

    # Ensure threads don't go out of bounds
    if x > 0 and x < input_image.shape[0] - 1 and y > 0 and y < input_image.shape[1] - 1:
        gx = 0.0
        gy = 0.0

        # Apply Sobel operator
        for i in range(-1, 2):
            for j in range(-1, 2):
                gx += input_image[x + i, y + j] * sobel_x[i + 1, j + 1]
                gy += input_image[x + i, y + j] * sobel_y[i + 1, j + 1]

        # Compute gradient magnitude and store result
        output_image[x, y] = min(255, math.sqrt(gx ** 2 + gy ** 2))
