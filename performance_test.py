import time
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from src.image_processor import process_image
from src.utils.image_io import load_image

# CPU implementation of edge detection
def edge_detection_cpu(image):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    gx = cv2.filter2D(image, -1, kernel)
    return np.clip(np.abs(gx), 0, 255).astype(np.uint8)

    
def save_results_to_csv(cpu_time, gpu_time, filename="performance_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Implementation", "Time (seconds)"])
        writer.writerow(["CPU", f"{cpu_time:.4f}"])
        writer.writerow(["GPU", f"{gpu_time:.4f}"])
    print(f"Results saved to {filename}")

def plot_results(cpu_time, gpu_time):
    implementations = ['CPU', 'GPU']
    times = [cpu_time, gpu_time]
    
    plt.bar(implementations, times, color=['blue', 'green'])
    plt.ylabel('Time (seconds)')
    plt.title('CPU vs GPU Performance')
    plt.show()
    
def test_performance():
    input_path = "data/input_image.jpg"
    print("Loading image...")
    image = load_image(input_path)

    print("Running CPU edge detection...")
    start_cpu = time.time()
    cpu_result = edge_detection_cpu(image)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    print("Running GPU edge detection...")
    start_gpu = time.time()
    gpu_result = process_image(image, "basic")
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"CPU Time: {cpu_time:.4f} seconds")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    save_results_to_csv(cpu_time, gpu_time)
    plot_results(cpu_time, gpu_time)
 

def main():
    test_performance()

if __name__ == "__main__":
    main()
