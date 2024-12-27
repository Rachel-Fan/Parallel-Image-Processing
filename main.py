import sys
import os
import cv2 
import matplotlib.pyplot as plt

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "src")
sys.path.insert(0, project_root)

from src.utils.process_images import process_images

if __name__ == "__main__":
    # Redirect input image to be loaded from file
    input_image_path = os.path.join('data', 'input_image.jpg')
    input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    hist_eq_img, edge_img_original, edge_img_hist_eq = process_images(input_img)

    output_folder = os.path.join('data', 'output')
    os.makedirs(output_folder, exist_ok=True)
    

    plt.figure(figsize=(15, 10))
    # First row
    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("Edge Detection on Original Image")
    plt.imshow(edge_img_original, cmap='gray')
    # Second row
    plt.subplot(2, 2, 3)
    plt.title("Histogram Equalized")
    plt.imshow(hist_eq_img, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title("Edge Detection on Equalized Image")
    plt.imshow(edge_img_hist_eq, cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "GPU_image_processing.png"))
    plt.show()
