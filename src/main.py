from image_processor import process_image
from utils.image_io import load_image, save_image

def main():
    input_path = "data/input_image.jpg"
    output_path = "data/output_image.jpg"

    print("Loading image...")
    image = load_image(input_path)

    print("Processing image with CUDA...")
    processed_image = process_image(image, "basic")

    print("Saving processed image...")
    save_image(output_path, processed_image)

    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    main()
