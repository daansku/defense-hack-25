import os
import json
import time
from datetime import datetime
from ultralytics import YOLO

# Configuration
WATCH_FOLDER = "C:/Transfers"
MODEL_PATH = "yolov8n.pt"

# Load YOLO model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Track processed files
processed = set()

print(f"Monitoring folder: {WATCH_FOLDER}")
print("Press Ctrl+C to stop\n")

while True:
    try:
        # Get all image files in folder
        files = [f for f in os.listdir(WATCH_FOLDER) 
                if f.endswith(('.jpg', '.jpeg', '.png')) and "cropped" in f]
        
        # Process new files
        for filename in files:
            filepath = os.path.join(WATCH_FOLDER, filename)
            
            # Skip if already processed
            if filepath in processed:
                continue
            
            processed.add(filepath)
            print(f"Processing: {filename}")
            
            # Extract node_id (last char before extension)
            node_id = int(filename.split('.')[0][-1]) if filename.split('.')[0][-1].isdigit() else 1
            
            # Run YOLO detection
            results = model(filepath, verbose=False)
            
            # Collect detections
            detections = []
            object_counts = {}
            
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        "class_name": class_name if class_name == "person" else "",
                        "confidence": confidence
                    })
                    
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Create JSON output
            output = {
                "timestamp": datetime.now().isoformat(),
                "node_id": node_id,
                "image_path": filepath.replace("\\", "/"),
                "total_detections": len(detections),
                "object_counts": object_counts,
                "detections": detections
            }
            
            # Print and save
            print(json.dumps(output, indent=2))
            
            # Save to file
            json_file = f"detections/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"Saved to: {json_file}\n")
        
        time.sleep(1)  # Check every second
        
    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)