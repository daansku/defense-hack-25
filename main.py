import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('motion_detection.log')
    ]
)


def find_bounding_box(diff_map: np.ndarray, min_size=20):
    """
    Find bounding box of motion in difference map
    
    Args:
        diff_map: Binary difference map
        min_size: Minimum box size to consider
        
    Returns:
        tuple: (x1, y1, x2, y2) or None if no significant motion
    """
    # Get non-zero points (where motion occurred)
    y_coords, x_coords = np.nonzero(diff_map)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
        
    # Get bounding box
    x1, x2 = np.min(x_coords), np.max(x_coords)
    y1, y2 = np.min(y_coords), np.max(y_coords)
    
    # Add small padding
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(diff_map.shape[1] - 1, x2 + pad)
    y2 = min(diff_map.shape[0] - 1, y2 + pad)
    
    # Check if box is too small
    if (x2 - x1 < min_size) or (y2 - y1 < min_size):
        return None
        
    return (x1, y1, x2, y2)

def MSE(image1: np.ndarray, image2: np.ndarray, thresh=0.03):
    """
    Returns the MSE of greyscale images image1 and image2 and bounding box if motion detected

    Args:
        image1, image2: np.ndarray of shape (H, W) with pixel values in [0, 1] (grayscale)
        thresh: threshold for motion detection

    Returns:
        tuple: (image2 if motion detected else None, mse value, bounding box or None, diff_visualization)
    """
    # Calculate absolute difference and apply Gaussian blur to reduce noise
    diff = np.abs(image1 - image2)
    diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    mse = np.mean(diff_blurred ** 2)
    
    # Create visualization of the difference
    # Scale to 0-255 for better visibility
    diff_vis = (diff_blurred * 255).astype(np.uint8)
    # Apply color map for better visualization
    diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
    
    if mse > thresh:
        # Create binary difference map with higher local threshold
        diff_map = (diff_blurred > thresh).astype(np.uint8)
        
        # Apply morphological operations to clean up the motion mask
        kernel = np.ones((5,5), np.uint8)
        diff_map = cv2.morphologyEx(diff_map, cv2.MORPH_OPEN, kernel)  # Remove noise
        diff_map = cv2.morphologyEx(diff_map, cv2.MORPH_CLOSE, kernel)  # Fill holes
        diff_map = cv2.dilate(diff_map, kernel, iterations=1)  # Expand slightly
        
        # Find bounding box
        bbox = find_bounding_box(diff_map)
        
        # Draw the binary mask on the visualization
        diff_vis = cv2.addWeighted(diff_vis, 0.7, cv2.cvtColor(diff_map * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        return image2, mse, bbox, diff_vis
    
    return None, mse, None, diff_vis


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


def crop_and_compress(frame: np.ndarray, bbox, quality=30) -> np.ndarray:
    """
    Crop frame to bounding box and compress
    
    Args:
        frame: Full frame
        bbox: Tuple (x1, y1, x2, y2)
        quality: JPEG compression quality (0-100)
        
    Returns:
        Compressed cropped frame
    """
    try:
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = bbox
        
        # Validate bbox coordinates
        if not all(isinstance(x, (int, np.integer)) for x in [x1, y1, x2, y2]):
            logging.error("Invalid bbox coordinates type")
            return None
            
        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            logging.error("Bbox coordinates out of bounds")
            return None
            
        cropped = frame[y1:y2, x1:x2]
        
        # Compress using JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed = cv2.imencode('.jpg', cropped, encode_param)
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        
        if decompressed is None:
            logging.error("Failed to compress/decompress image")
            return None
            
        logging.info(f"Cropped and compressed image from {frame.shape} to {decompressed.shape}")
        return decompressed
        
    except Exception as e:
        logging.error(f"Error in crop_and_compress: {str(e)}")
        return None

def save_frame(frame, bbox=None, output_dir="captured_images", is_visualization=False) -> str:
    """Save a frame and return the filename"""
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "viz_" if is_visualization else "capture_"
    
    if bbox is not None and not is_visualization:
        # Save cropped and compressed version
        compressed = crop_and_compress(frame, bbox)
        if compressed is not None:
            frame = compressed
            
    filename = os.path.join(output_dir, f'{prefix}{timestamp}.jpg')
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")
    return filename


def delete_oldest_image(folder="motion_detected", keep_count=50):
    """Keep only the specified number of most recent motion detection images"""
    folder = Path(folder)
    if not folder.exists():
        return
        
    imgs = sorted(folder.glob("*.jpg"), key=os.path.getmtime)
    while len(imgs) > keep_count:
        os.remove(imgs[0])
        print(f"Deleted old motion image: {imgs[0]}")
        imgs.pop(0)


def main():
    try:
        # Create output directory
        Path("captured_images").mkdir(exist_ok=True)
        logging.info("Starting motion detection")
        
        # 0 for webcam 1 for logitech 
        cam = cv2.VideoCapture(1)
        
        if not cam.isOpened():
            logging.error("Could not open camera")
            return
            
        logging.info("Camera initialized successfully")
        
        frame_idx = 0

        while True:
            ret, frame = cam.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Display the current frame
            cv2.imshow("Motion Detection", frame)

            # Process every 5th frame
            if frame_idx % 5 == 0:
                # Convert current frame to grayscale
                current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
                
                # Compare with previous frame if we have one
                if 'prev_frame_gray' in locals():
                    result, mse_value, bbox, diff_vis = MSE(prev_frame_gray, current_frame_gray)
                    
                    # Always show the difference visualization (but don't save)
                    cv2.imshow("Motion Detection - Difference", diff_vis)
                    
                    if result is not None and bbox is not None:
                        x1, y1, x2, y2 = bbox
                        print(f"Motion detected! MSE: {mse_value:.6f}")
                        print(f"Motion box: ({x1}, {y1}) to ({x2}, {y2})")
                        
                        # Draw rectangle on frame
                        frame_with_box = frame.copy()
                        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Save the frame with bounding box
                        save_frame(frame_with_box, output_dir="motion_detected")
                
                # Update previous frame
                prev_frame_gray = current_frame_gray.copy()

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