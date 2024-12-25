from kernels.edge_detection_kernel import get_basic_kernel
from kernels.custom_kernel import get_custom_kernel
from utils.gpu_memory import allocate_memory, free_memory
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np

def process_image(image, kernel_type="basic"):
    height, width = image.shape
    kernel_map = {
        "basic": get_basic_kernel,
        "custom": get_custom_kernel,
    }

    if kernel_type not in kernel_map:
        raise ValueError(f"Kernel type '{kernel_type}' is not supported.")

    kernel_code = kernel_map[kernel_type]()
    mod = SourceModule(kernel_code)
    func = mod.get_function("process_image")

    # Allocate memory on GPU
    input_gpu, output_gpu = allocate_memory(image)

    # Copy data to GPU
    cuda.memcpy_htod(input_gpu, image)

    # Configure and launch kernel
    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    func(input_gpu, output_gpu, np.int32(width), np.int32(height),
         block=block_size, grid=grid_size)

    # Retrieve processed image
    result = np.empty_like(image)
    cuda.memcpy_dtoh(result, output_gpu)

    # Free GPU memory
    free_memory(input_gpu, output_gpu)

    return result
