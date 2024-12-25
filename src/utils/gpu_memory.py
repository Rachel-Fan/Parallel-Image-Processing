import pycuda.driver as cuda

def allocate_memory(image):
    input_gpu = cuda.mem_alloc(image.nbytes)
    output_gpu = cuda.mem_alloc(image.nbytes)
    return input_gpu, output_gpu

def free_memory(*buffers):
    for buf in buffers:
        buf.free()
