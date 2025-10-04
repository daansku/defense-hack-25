# Defense Hack 2025 - Motion Detection and Visualization System

This project implements a real-time motion detection and visualization system using computer vision and object detection. The system consists of two main components:

1. Motion detection and object recognition pipeline
2. Interactive battlefield visualization demo

Briefly about the context:
The motion detection is performed at edge sensors placed along tree lines and fields. An edge sensor is a device that has at least a camera, a mini processor, a battery and a radio transmitter. Whenever motion is detected, the device segments the moving object and sends the image over the network of other edge sensors until the image reaches the main node which is, for example, a laptop that works as a hub for all edge sensors in its reach. The main node runs a classification model that turns the images into JSON files that list all classified objects. These JSON files are fed into a pipeline that generates FRAGO reports.

## Project Structure

```
defense-hack-25/
├── detections/               # JSON files containing the details about each detection
├── motion_detected/         # Captured and cropped motion detection images
├── server/                  # Everything that should happen on the server
│   ├── battlefield_ui.py    # Interactive visualization demo
│   ├── detection.py        # Object detection using YOLOv8
│   └── nodes.json          # Sample nodes for visualization demo
├── main.py                 # Main motion detection script
├── floormap.png           # Floor plan image for visualization demo
└── requirements.txt       # Python dependencies
```

## Features

- Real-time motion detection using OpenCV
- Object detection using YOLOv8
- Interactive demo with Pygame
- Support for multiple detection nodes
- Detection history and visualization
- Automatic image cropping and saving

## Installation

1. Clone the repository:

```bash
gh repo clone daansku/defense-hack-25
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The system can be run in two modes:

### 1. Motion Detection Pipeline

Run the motion detection and object recognition:

```bash
python main.py
```

This will:

- Initialize the camera
- Start motion detection
- Save detected motion images to `motion_detected/`
- Run object detection on motion events
- Save detection results to `detections/`

### 2. Battlefield Visualization Demo

Run the visualization interface:

```bash
python server/battlefield_ui.py
```

This will:

- Display the floor plan with detection nodes
- Show real-time detection updates (assuming cameras are connected)
- Visualize detected objects and their locations
- Provide an interactive interface for monitoring

### Visualization Controls

- Click on a node to show/hide its detection information
- Click the "Hide/Show Messages" button to toggle detection message history
- Press 'q' to quit the visualization

## Configuration

- `server/nodes.json`: Configure node positions and descriptions
- `main.py`: Adjust motion detection parameters (CAMERA_ID, PIXEL_DIFF_THRESH, INITIAL_FRAME_SKIP, etc.)
- `server/detection.py`: Modify object detection settings (confidence threshold, allowed classes)

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- Pygame
- Additional dependencies in requirements.txt

## Notes

- The system requires a camera for motion detection (configurable in main.py: CAMERA_ID). Set 0 for the build-in laptop webcam and 1 for the external camera (e.g. Logitech)
- Detection results are stored in JSON format for easy integration
- The visualization demo supports real-time updates from multiple detection nodes
