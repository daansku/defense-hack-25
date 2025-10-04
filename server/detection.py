import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from ultralytics import YOLO
import numpy as np

# Load model with adjusted NMS parameters
model = YOLO("yolov8n.pt")
model.conf = 0.5  # Higher confidence threshold
model.iou = 0.7   # Higher IoU threshold for NMS

allowed_classes = ["person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"backpack",
"handbag",
"laptop",
"cell phone"]

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

def calculate_box_area(box) -> float:
    """Calculate the area of a bounding box."""
    coords = box.xyxy[0].tolist()
    width = coords[2] - coords[0]
    height = coords[3] - coords[1]
    return width * height

def calculate_box_overlap(box1, box2) -> float:
    """Calculate the overlap ratio between two boxes."""
    coords1 = box1.xyxy[0].tolist()
    coords2 = box2.xyxy[0].tolist()
    
    # Calculate intersection
    x1 = max(coords1[0], coords2[0])
    y1 = max(coords1[1], coords2[1])
    x2 = min(coords1[2], coords2[2])
    y2 = min(coords1[3], coords2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)
    
    # Return overlap ratio relative to the smaller box
    return intersection / min(area1, area2)

def merge_overlapping_detections(boxes, names, overlap_threshold: float = 0.7) -> List:
    """Merge overlapping detections of the same class."""
    merged_detections = []
    used_boxes = set()
    
    for i, box1 in enumerate(boxes):
        if i in used_boxes:
            continue
            
        class_id1 = int(box1.cls[0])
        class_name1 = names[class_id1]
        
        if class_name1 not in allowed_classes:
            continue
        
        current_group = [box1]
        used_boxes.add(i)
        
        # Look for overlapping boxes of the same class
        for j, box2 in enumerate(boxes):
            if j in used_boxes:
                continue
                
            class_id2 = int(box2.cls[0])
            class_name2 = names[class_id2]
            
            if class_name1 == class_name2 and calculate_box_overlap(box1, box2) > overlap_threshold:
                current_group.append(box2)
                used_boxes.add(j)
        
        # If we found overlapping boxes, merge them by taking the one with highest confidence
        if current_group:
            best_box = max(current_group, key=lambda x: float(x.conf[0]))
            merged_detections.append((best_box, class_name1))
    
    return merged_detections

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
    
    # Merge overlapping detections
    merged_detections = merge_overlapping_detections(boxes, names)
    
    # Format results
    detections = []
    object_counts = {}
    
    for box, class_name in merged_detections:
        # Update object counts
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Add detection details
        detection = format_detection_for_json(box, class_name)
        detections.append(detection)
    
    # Create output structure
    output = {
        "timestamp": timestamp,
        "node_id": 1, # TODO: Get actual node id from image path
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