import cv2
import os
import time
from datetime import datetime
from typing import Callable, Optional, List

class CameraCapture:
    def __init__(self, camera_id: int = 1, capture_interval: int = 5, max_images: int = 2):
        self.output_dir = 'captured_images'
        self.camera_id = camera_id
        self.capture_interval = capture_interval
        self.max_images = max_images
        self.image_files: List[str] = []
        self.last_capture_time = time.time()
        self.camera = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def cleanup_old_images(self):
        """Keep only the most recent images based on max_images setting"""
        while len(self.image_files) > self.max_images:
            file_to_remove = self.image_files.pop(0)  # Remove oldest image
            try:
                os.remove(file_to_remove)
                print(f"Removed old image: {file_to_remove}")
            except OSError as e:
                print(f"Error removing file {file_to_remove}: {e}")

    def save_frame(self, frame) -> str:
        """Save a frame and return the filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f'capture_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        self.image_files.append(filename)
        print(f"Image saved: {filename}")
        self.cleanup_old_images()
        return filename

    def start_capture(self, process_image_callback: Optional[Callable[[str, cv2.Mat], None]] = None):
        """
        Start the camera capture loop
        
        Args:
            process_image_callback: Optional callback function that takes filename and frame as arguments
                                  for additional image processing
        """
        self.camera = cv2.VideoCapture(self.camera_id)
        
        try:
            while True:
                ret, frame = self.camera.read()
                current_time = time.time()
                
                if ret:
                    # Show the feed
                    cv2.imshow('Camera Feed', frame)
                    
                    # Check if capture_interval seconds have passed since last capture
                    if current_time - self.last_capture_time >= self.capture_interval:
                        filename = self.save_frame(frame)
                        self.last_capture_time = current_time
                        
                        # Call the callback function if provided
                        if process_image_callback:
                            process_image_callback(filename, frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

    def get_latest_images(self) -> List[str]:
        """Return list of current image files, newest last"""
        return self.image_files.copy()

# Example usage:
def process_image(filename: str, frame: cv2.Mat):
    """Example callback function for image processing"""
    # Add your image processing code here
    print(f"Processing image: {filename}")
    # Example: You could analyze the frame here
    # Your processing code goes here

if __name__ == "__main__":
    # Create camera capture instance
    camera = CameraCapture(camera_id=1, capture_interval=1, max_images=2)
    
    # Start capture with optional processing callback
    camera.start_capture(process_image_callback=process_image)