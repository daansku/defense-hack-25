import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
import os


def MSE(image1: np.ndarray, image2: np.ndarray, thresh=0.1):
    """
    Returns the MSE of greyscale images image1 and image2

    Args:
        image1, image2: np.ndarray of shape (H, W) with pixel values in [0, 1] (grayscale)
        thresh: threshold for motion detection

    Returns:
        tuple: (image2 if motion detected else None, mse value)
    """
    mse = np.mean((image1 - image2) ** 2)

    if mse > thresh:
        return image2, mse
    
    return None, mse


def convert_to_greyscale(image_path: str) -> np.ndarray:
    """
    Converts an RGB image to greyscale

    Args:
        image_path: path to the image file

    Returns:
        np.ndarray of shape (H, W) with pixel values in [0, 1] (grayscale)
    """
    img = Image.open(image_path)
    gray = img.convert('L')
    arr = np.array(gray) / 255.0

    return arr


def save_frame(frame, output_dir="captured_images") -> str:
    """Save a frame and return the filename"""
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f'capture_{timestamp}.jpg')
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")
    return filename


def delete_oldest_image(folder="captured_images"):
    """Keep only the 2 most recent images"""
    folder = Path(folder)
    imgs = sorted(folder.glob("*.jpg"), key=os.path.getmtime)
    
    while len(imgs) > 2:
        os.remove(imgs[0])
        print(f"Deleted: {imgs[0]}")
        imgs.pop(0)


def main():
    # Create output directory
    Path("captured_images").mkdir(exist_ok=True)
    
    # 0 for webcam 1 for logitech 
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_idx = 0

    try:
        while True:
            ret, frame = cam.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Display the current frame
            cv2.imshow("Motion Detection", frame)

            # Process every 30th frame
            if frame_idx % 30 == 0:
                save_frame(frame)
                
                folder = Path("captured_images/")
                imgs = []

                # Load all saved images
                for file in sorted(folder.glob("*.jpg"), key=os.path.getmtime):
                    img = convert_to_greyscale(str(file))
                    imgs.append(img)

                print(f"Images in folder: {len(imgs)}")

                # Compare the two most recent images
                if len(imgs) >= 2:
                    result, mse_value = MSE(imgs[-2], imgs[-1])
                    if result is not None:
                        print(f"Motion detected! MSE: {mse_value:.6f}")
                    else:
                        print(f"No motion. MSE: {mse_value:.6f}")

                # Keep only 2 most recent images
                if len(imgs) > 2:
                    delete_oldest_image()

            frame_idx += 1
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
                
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()