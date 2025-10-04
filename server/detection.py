# Outputs the detections from the images to /detections folder

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def get_latest_image(directory: Union[str, Path]) -> Optional[Path]:
    """
    Get the path of the most recently added image file in the specified directory.
    
    Args:
        directory: Path to the directory to search in
        
    Returns:
        Path to the latest image file or None if no images found
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return None
        
    # List all files and get their creation times
    files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        files.extend(directory.glob(f'*{ext}'))
    
    if not files:
        print(f"No image files found in {directory}")
        return None
        
    # Get the most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return latest_file

def format_detection_for_json(box, class_name: str) -> Dict:
    """Format a single detection box into a JSON-compatible dictionary."""
    coords = box.xyxy[0].tolist()  # get box coordinates
    return {
        "class_name": class_name,
        "confidence": float(box.conf[0]),
    }

def detect_objects(image_path: str, save_json: bool = True, output_dir: Optional[str] = None) -> Dict:
    """
    Detect objects in an image and optionally save results to JSON.
    
    Args:
        image_path: Path to the image file
        save_json: Whether to save results to a JSON file
        output_dir: Directory to save JSON output (defaults to 'detections' in current dir)
    
    Returns:
        Dictionary containing detection results
    """
    results = model(image_path)
    result = results[0]
    boxes = result.boxes
    names = result.names
    
    # Create timestamp for the detection
    timestamp = datetime.now().isoformat()
    
    # Format results
    detections = []
    object_counts = {}
    
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        
        # Update object counts
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Add detection details
        detection = format_detection_for_json(box, class_name)
        detections.append(detection)
    
    # Create output structure
    output = {
        "timestamp": timestamp,
        "image_path": image_path,
        "total_detections": len(detections),
        "object_counts": object_counts,
        "detections": detections
    }
    
    # Save to JSON file if requested
    if save_json:
        output_dir = output_dir or "detections"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create filename based on timestamp
        filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = Path(output_dir) / filename
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
            print(f"Detection results saved to {output_path}")
    
    return output

if __name__ == "__main__":
    # Get the latest image from motion_detected folder
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