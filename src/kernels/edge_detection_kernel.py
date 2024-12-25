def get_edge_detection_kernel():
    return """
    __global__ void edgeDetectionKernel(unsigned char *input, unsigned char *output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int idx = y * width + x;

            int gx = -1 * input[idx - 1 - width] + 1 * input[idx + 1 - width]
                     -2 * input[idx - 1]        + 2 * input[idx + 1]
                     -1 * input[idx - 1 + width] + 1 * input[idx + 1 + width];
            gx = abs(gx);
            output[idx] = min(max(gx, 0), 255);
        }
    }
    """