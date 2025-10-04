import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
import os
import logging
from server.detection import get_latest_image, detect_objects

PIXEL_DIFF_THRESH = 0.3   # Threshold for pixel difference (0-1 range)
INITIAL_FRAME_SKIP = 30
CAMERA_ID = 1
KEEP_PREV_MOTION_PICS = False
KEEP_PREV_CAPTURE_PICS = False
COMPRESS_QUALITY = 30  
MAX_AFTER_DET_FRAMES = 5  # Number of frames to process after detection
AFTER_DET_RATE = 20

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
        logging.warning("No non-zero pixels found in diff_map")
        return None
    
    logging.info(f"Found {len(x_coords)} motion pixels")
        
    # Get bounding box
    x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
    y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
    
    logging.info(f"Raw bbox before padding: x=[{x1}, {x2}], y=[{y1}, {y2}]")
    logging.info(f"Image shape: {diff_map.shape}")
    
    # Add small padding
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(diff_map.shape[1] - 1, x2 + pad)
    y2 = min(diff_map.shape[0] - 1, y2 + pad)
    
    logging.info(f"Final bbox after padding: x=[{x1}, {x2}], y=[{y1}, {y2}]")
    
    # Check if box is too small
    if (x2 - x1 < min_size) or (y2 - y1 < min_size):
        logging.warning(f"Bbox too small: width={x2-x1}, height={y2-y1}")
        return None
    
    # Check if box covers most of the image (likely a false positive)
    image_area = diff_map.shape[0] * diff_map.shape[1]
    bbox_area = (x2 - x1) * (y2 - y1)
    coverage = bbox_area / image_area
    
    if coverage > 0.9:  # If bbox covers >90% of image
        logging.warning(f"Bbox covers {coverage*100:.1f}% of image - likely false positive")
        return None
        
    return (x1, y1, x2, y2)


def visualize_binary(diff_map: np.ndarray) -> None:
    """Visualize binary difference map"""
    vis = (diff_map * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_BONE)
    cv2.imshow("Binary Difference Map", vis)


def detect_motion(image1: np.ndarray, image2: np.ndarray, thresh=0.1, save_crop=False, 
                  original_frame=None, output_dir="motion_detected"):
    """
    Detect motion by comparing pixel differences between two RGB/BGR images
    
    Args:
        image1, image2: np.ndarray of shape (H, W, 3) with pixel values in [0, 1] (RGB/BGR)
        thresh: threshold for pixel difference detection
        save_crop: whether to save the cropped bounding box image
        original_frame: original color frame (BGR) to crop from, if None uses image2
        output_dir: directory to save cropped images
        
    Returns:
        tuple: (image2 if motion detected else None, num_changed_pixels, bounding box or None, diff_visualization)
    """
    # Calculate absolute difference per channel
    diff = np.abs(image1 - image2)
    
    # Average across color channels to get single difference value per pixel
    diff_avg = np.mean(diff, axis=2)
    
    # Apply Gaussian blur to reduce noise
    diff_blurred = cv2.GaussianBlur(diff_avg, (5, 5), 0)
    
    # Create binary difference map where pixels exceed threshold
    diff_map = (diff_blurred > thresh).astype(np.uint8)
    
    # Count number of changed pixels
    num_changed_pixels = np.sum(diff_map)
    
    # Create visualization of the difference
    diff_vis = (diff_blurred * 255).astype(np.uint8)
    diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
    
    if num_changed_pixels > 0:
        logging.info(f"Changed pixels: {num_changed_pixels}")
        
        # Apply morphological operations to clean up the motion mask
        kernel = np.ones((5, 5), np.uint8)
        diff_map = cv2.morphologyEx(diff_map, cv2.MORPH_OPEN, kernel)   # Remove noise
        diff_map = cv2.morphologyEx(diff_map, cv2.MORPH_CLOSE, kernel)  # Fill holes
        diff_map = cv2.dilate(diff_map, kernel, iterations=1)           # Expand slightly
        
        # Find bounding box
        bbox = find_bounding_box(diff_map)

        visualize_binary(diff_map)
        
        # Draw the binary mask on the visualization
        diff_vis = cv2.addWeighted(diff_vis, 0.7, cv2.cvtColor(diff_map * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Save cropped image if requested and bbox is valid
        if save_crop and bbox is not None:
            
            # Determine which frame to crop from
            if original_frame is not None:
                frame_to_crop = original_frame
            else:
                # Convert back to 0-255 range for saving
                frame_to_crop = (image2 * 255).astype(np.uint8)
            
            # Crop the bounding box
            cropped = crop_to_grayscale(frame_to_crop, bbox)
            
            # Save the cropped image
            Path(output_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            crop_filename = os.path.join(output_dir, f"cropped_{timestamp}.jpg")
            cv2.imwrite(crop_filename, cropped)
            logging.info(f"Saved cropped image: {crop_filename} (size: {cropped.shape})")
        
        return image2, num_changed_pixels, bbox, diff_vis
    
    return None, num_changed_pixels, None, diff_vis


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


def crop_to_grayscale(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop frame to bounding box and convert to grayscale"""
    if bbox is None:
        return None
        
    x1, y1, x2, y2 = bbox
    
    # Validate and crop
    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
        logging.error("Bbox coordinates out of bounds")
        return None
        
    cropped = frame[y1:y2, x1:x2]
    
    # Convert to grayscale if needed
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    return cropped


def save_frame(frame, bbox=None, output_dir="captured_images", is_visualization=False) -> str:
    """Save a frame and return the filename"""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "viz_" if is_visualization else "capture_"
    
    if bbox is not None and not is_visualization:
        compressed = crop_to_grayscale(frame, bbox)
        if compressed is not None:
            frame = compressed
            
    filename = os.path.join(output_dir, f'{prefix}{timestamp}.jpg')
    if len(frame.shape) == 3:  # Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename, gray_frame)
    else:
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


def delete_all_images(folder_path="captured_images"):
    """Delete all image files in a folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        logging.warning(f"Folder {folder_path} does not exist")
        return
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    deleted_count = 0
    
    for ext in image_extensions:
        for img_file in folder.glob(ext):
            try:
                img_file.unlink()
                deleted_count += 1
            except Exception as e:
                logging.error(f"Error deleting {img_file.name}: {e}")
    
    logging.info(f"Deleted {deleted_count} images from {folder_path}")


def process_folder_images(folder_path="captured_images"):
    """
    Process images from folder: use oldest as background, newest for motion detection
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        logging.error(f"Folder {folder_path} does not exist")
        return None
    
    # Get all image files, sorted by modification time
    image_files = sorted(folder.glob("*.jpg"), key=os.path.getmtime)
    
    # Filter out visualization files
    image_files = [f for f in image_files if not f.name.startswith("viz_")]
    
    if len(image_files) < 2:
        logging.warning(f"Need at least 2 images in {folder_path}, found {len(image_files)}")
        return None
    
    # Oldest image as background
    background_path = image_files[0]
    # Newest image for motion detection
    newest_path = image_files[-1]
    
    logging.info(f"Background: {background_path.name}")
    logging.info(f"Newest image: {newest_path.name}")
    
    # Load images
    background_frame = cv2.imread(str(background_path))
    newest_frame = cv2.imread(str(newest_path))
    
    if background_frame is None or newest_frame is None:
        logging.error("Failed to load images")
        return None
    
    # Convert to 0-1 range (keep RGB/BGR)
    background_rgb = background_frame.astype(np.float32) / 255.0
    newest_rgb = newest_frame.astype(np.float32) / 255.0
    
    # Detect motion using pixel difference
    result, num_changed, bbox, diff_vis = detect_motion(
        background_rgb, 
        newest_rgb,
        thresh=PIXEL_DIFF_THRESH,
        save_crop=True, 
        original_frame=newest_frame,
        output_dir="motion_detected"
    )
    
    if result is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        logging.info(f"Motion detected! Changed pixels: {num_changed}")
        logging.info(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Draw rectangle on newest frame
        frame_with_box = newest_frame.copy()
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save visualization
        save_frame(frame_with_box, output_dir="motion_detected")
        save_frame(diff_vis, output_dir="motion_detected", is_visualization=True)
        
        return background_rgb, newest_frame, newest_rgb, bbox
    else:
        logging.info(f"No significant motion detected. Changed pixels: {num_changed}")
        return None


def detect_obj():
    motion_dir = Path("motion_detected")
    latest_image = get_latest_image(motion_dir)
    
    if latest_image is None:
        print("No images found to process")
        exit(1)
        
    print(f"Processing latest image: {latest_image}")
    results = detect_objects(str(latest_image))
    
    # Print summary
    print("\nDetection Summary:")
    print(f"Total detections: {results['total_detections']}")
    print("\nObject counts:")
    for obj, count in results['object_counts'].items():
        print(f"{obj}: {count}")


def img_to_bytestr(path) -> bytes:
    """Read an image and encode as JPEG bytes"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)   
    success, encoded = cv2.imencode(".jpg", img)   
    if not success:
        raise ValueError("Could not encode image")
    return encoded.tobytes()  


def main():
    try:
        # Create output directories
        Path("motion_detected/").mkdir(exist_ok=True)
        logging.info("Starting motion detection")
        
        if not KEEP_PREV_MOTION_PICS:
            delete_all_images("motion_detected")
        if not KEEP_PREV_CAPTURE_PICS:
            delete_all_images("captured_images")

        # Process existing images in folder
        result = process_folder_images("captured_images")
        
        if result is None:
            logging.info("No motion detected or insufficient images. Starting camera capture...")
        
        # Open camera (0 for webcam, 1 for logitech)
        cam = cv2.VideoCapture(CAMERA_ID)
        
        if not cam.isOpened():
            logging.error("Could not open camera")
            return
            
        logging.info("Camera initialized successfully")
        frame_idx = 0
        after_det_frames = 0 
        FRAME_SKIP = INITIAL_FRAME_SKIP


        
        while True:
            # Process every Nth frame
            if frame_idx % FRAME_SKIP == 0:

                if after_det_frames > MAX_AFTER_DET_FRAMES:
                    FRAME_SKIP = INITIAL_FRAME_SKIP
                    after_det_frames = 0

                ret, frame = cam.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                cv2.imshow("Motion Detection", frame)

                # Convert current frame to 0-1 range (keep RGB/BGR)
                current_frame_rgb = frame.astype(np.float32) / 255.0
                
                # Compare with previous frame if we have one
                if 'prev_frame_rgb' in locals():
                    # Only save crops on the last frame (when after_det_frames == MAX_AFTER_DET_FRAMES)
                    result, num_changed, bbox, diff_vis = detect_motion(
                        prev_frame_rgb, 
                        current_frame_rgb,
                        thresh=PIXEL_DIFF_THRESH,
                        save_crop=False,  # Don't save crops during motion tracking
                        original_frame=frame,
                        output_dir="motion_detected"
                    )
                    
                    # Always show the difference visualization
                    cv2.imshow("Motion Detection - Difference", diff_vis)
                    
                    if result is not None and bbox is not None:
                        x1, y1, x2, y2 = bbox
                        print(f"Motion detected! Changed pixels: {num_changed}")
                        print(f"Motion box: ({x1}, {y1}) to ({x2}, {y2})")
                        
                        # Draw rectangle on frame
                        frame_with_box = frame.copy()
                        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Only save the frame if it's the last one
                        if after_det_frames == MAX_AFTER_DET_FRAMES:
                            # Save both the frame with bounding box and the cropped version
                            save_frame(frame_with_box, output_dir="motion_detected")
                            # Save cropped version by calling detect_motion with save_crop=True
                            detect_motion(
                                prev_frame_rgb, 
                                current_frame_rgb,
                                thresh=PIXEL_DIFF_THRESH,
                                save_crop=True,
                                original_frame=frame,
                                output_dir="motion_detected"
                            )
                            detect_obj()
                            print(img_to_bytestr(get_latest_image("motion_detected")))
                            exit(0)

                        if FRAME_SKIP == INITIAL_FRAME_SKIP:
                            FRAME_SKIP = AFTER_DET_RATE

                        if FRAME_SKIP == AFTER_DET_RATE:
                            after_det_frames += 1
                
                # Update previous frame
                prev_frame_rgb = current_frame_rgb.copy()

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