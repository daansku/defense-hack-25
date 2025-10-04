import numpy as np
from typing import Optional
from PIL import Image

def MSE(image1: np.ndarray, image2: np.ndarray, thresh=0.1) -> Optional[np.ndarray]:

    """
    returns the mse of greyscale images image1 and image2

    args:
    image1, image2: np.ndarray of shape (H, W) with pixel values in [0, 1] (grayscale)

    returns:
    Optional[float]: the mean squared error between image1 and image2

    """

    mse = np.mean((image1 - image2) ** 2)

    if mse > thresh:
        return image2
    
    return mse

def convert_to_greyscale(image_path: str) -> np.ndarray:

    """
    converts an RGB image to greyscale

    args:
    image: np.ndarray of shape (H, W, 3) with pixel values in [0, 1] (RGB)

    returns:
    np.ndarray of shape (H, W) with pixel values in [0, 1] (grayscale)

    """

    img = Image.open(image_path)
    gray = img.convert('L')
    arr = np.array(gray) / 255.0

    return arr

image_path1 = "photo_2025-10-04_10-39-50.jpg"
image_path2 = "photo_2025-10-04_10-45-03.jpg"

img1 = convert_to_greyscale(image_path1)
img2 = convert_to_greyscale(image_path2)

result = MSE(img1, img2, thresh=0.1)
print(result)





