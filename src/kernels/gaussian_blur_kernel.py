def get_gaussian_blur_kernel():
    return """
    __global__ void gaussianBlurKernel(unsigned char *input, unsigned char *output, int width, int height, float *kernel, int kernelSize) {
        extern __shared__ unsigned char sharedMem[];
        int radius = kernelSize / 2;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int localX = threadIdx.x + radius;
        int localY = threadIdx.y + radius;
        int sharedIdx = localY * (blockDim.x + 2 * radius) + localX;

        if (x < width && y < height) {
            sharedMem[sharedIdx] = input[y * width + x];
        }
    }
    """