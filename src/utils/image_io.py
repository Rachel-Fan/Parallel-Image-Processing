import cv2

def load_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return image

def save_image(filepath, image):
    cv2.imwrite(filepath, image)
