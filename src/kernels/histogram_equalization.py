from numba import cuda

@cuda.jit
def histogram_equalization(input_image, output_image, histogram, cdf_min, total_pixels):
    x, y = cuda.grid(2)

    if x < input_image.shape[0] and y < input_image.shape[1]:
        pixel_value = input_image[x, y]
        output_image[x, y] = int((histogram[pixel_value] - cdf_min) / (total_pixels - cdf_min) * 255)
