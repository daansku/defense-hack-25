import pygame
import json
import os
import time
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Initialize Pygame
pygame.init()

# Initialize font
pygame.font.init()
font = pygame.font.SysFont('JetBrains Mono', 16)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ROOM_WIDTH = 600
ROOM_HEIGHT = 400
NODE_RADIUS = 10
FLASH_DURATION = 1.0  # Duration of flash in seconds
DETECTION_CHECK_INTERVAL = 1.0  # How often to check for detections (seconds)
IMAGE_DISPLAY_SIZE = (200, 150)  # Size for the detection image thumbnail

# Colors
BLACK = (0, 0, 0)
DARK_GREY = (40, 40, 40)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Create the window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Battlefield UI")

# Global variables for flash control and UI state
last_detection_time = 0
is_flashing = False
last_detection_check = 0
active_detection = False
current_detection_image = None
current_image_path = None
current_detection_data = None
show_detection_info = False

class MotionDetectionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            global last_detection_time, is_flashing
            last_detection_time = time.time()
            is_flashing = True

def setup_motion_detection_observer():
    """Set up the file system observer for the motion_detected directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    motion_detected_dir = os.path.join(os.path.dirname(script_dir), "motion_detected")
    
    # Create motion_detected directory if it doesn't exist
    if not os.path.exists(motion_detected_dir):
        os.makedirs(motion_detected_dir)
    
    event_handler = MotionDetectionHandler()
    observer = Observer()
    observer.schedule(event_handler, motion_detected_dir, recursive=False)
    observer.start()
    return observer

def load_and_scale_image(image_path):
    """Load and scale an image to the display size"""
    try:
        image = pygame.image.load(image_path)
        return pygame.transform.scale(image, IMAGE_DISPLAY_SIZE)
    except (pygame.error, FileNotFoundError):
        return None

def check_active_detections():
    """Check if there are any active detections for node_id = 1"""
    global active_detection, last_detection_check, current_detection_image, current_image_path, current_detection_data
    
    current_time = time.time()
    # Only check periodically to avoid excessive file system operations
    if current_time - last_detection_check < DETECTION_CHECK_INTERVAL:
        return active_detection
    
    last_detection_check = current_time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detections_dir = os.path.join(os.path.dirname(script_dir), "detections")
    
    if not os.path.exists(detections_dir):
        active_detection = False
        current_detection_image = None
        current_image_path = None
        current_detection_data = None
        return False
    
    # Get all detection files
    detection_files = glob.glob(os.path.join(detections_dir, "detection_*.json"))
    
    # Get the most recent detection file
    latest_detection = None
    latest_time = 0
    
    # Check each detection file for node_id = 1
    for detection_file in detection_files:
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
                if data.get("node_id") == 1:
                    timestamp = time.mktime(time.strptime(data["timestamp"].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
                    if timestamp > latest_time:
                        latest_time = timestamp
                        latest_detection = data
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    
    if latest_detection:
        active_detection = True
        current_detection_data = latest_detection
        # Load the associated image if it's different from the current one
        image_path = os.path.join(os.path.dirname(script_dir), latest_detection["image_path"].replace('\\', '/'))
        if image_path != current_image_path:
            current_image_path = image_path
            current_detection_image = load_and_scale_image(image_path)
        return True
    
    active_detection = False
    current_detection_image = None
    current_image_path = None
    current_detection_data = None
    return False

def load_nodes():
    """Load nodes from the JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "nodes.json")
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def get_node_color():
    """Determine the node color based on flash state and active detections"""
    global is_flashing, last_detection_time
    
    # Check for active detections
    has_active_detections = check_active_detections()
    
    if is_flashing or has_active_detections:
        current_time = time.time()
        if is_flashing and current_time - last_detection_time > FLASH_DURATION:
            is_flashing = False
            # If there are active detections, keep flashing
            if not has_active_detections:
                return WHITE
        
        # Flash effect: alternate between red and white every 0.2 seconds
        return RED if int(current_time * 5) % 2 == 0 else WHITE
    
    return WHITE

def is_point_in_circle(point, center, radius):
    """Check if a point is inside a circle"""
    return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 <= radius ** 2

def draw_detection_info(screen, image_pos, detection_data):
    """Draw detection information below the image"""
    if not detection_data:
        return

    # Position for the text (below the image)
    text_x = image_pos[0]
    text_y = image_pos[1] + IMAGE_DISPLAY_SIZE[1] + 5

    # Create text surfaces for object counts
    object_counts = detection_data.get("object_counts", {})
    total_text = font.render(f"Total detections: {detection_data.get('total_detections', 0)}", True, WHITE)
    screen.blit(total_text, (text_x, text_y))

    # Display individual object counts
    y_offset = text_y + 20
    for obj_type, count in object_counts.items():
        count_text = font.render(f"{obj_type}: {count}", True, WHITE)
        screen.blit(count_text, (text_x, y_offset))
        y_offset += 20

def draw_battlefield():
    """Draw the battlefield with room and nodes"""
    # Fill background with black
    screen.fill(BLACK)
    
    # Calculate room position to center it
    room_x = (WINDOW_WIDTH - ROOM_WIDTH) // 2
    room_y = (WINDOW_HEIGHT - ROOM_HEIGHT) // 2
    
    # Draw dark grey room
    room_rect = pygame.Rect(room_x, room_y, ROOM_WIDTH, ROOM_HEIGHT)
    pygame.draw.rect(screen, DARK_GREY, room_rect)
    
    # Load and draw nodes
    data = load_nodes()
    node_pos = None
    
    for node in data["nodes"]:
        if node["id"] == 1:  # We only want node 1 for now
            # Store node position for image placement
            node_pos = (room_x + node["coordinates"]["x"], 
                       room_y + node["coordinates"]["y"])
            
            # Draw the node with dynamic color
            node_color = get_node_color()
            pygame.draw.circle(screen, node_color, node_pos, NODE_RADIUS)
            break
    
    # Draw detection image and info if toggled on
    if show_detection_info and current_detection_image and node_pos:
        # Position the image to the right of the node
        image_x = node_pos[0] + NODE_RADIUS + 10
        image_y = node_pos[1] - IMAGE_DISPLAY_SIZE[1] // 2
        
        # Draw a white border around the image
        border_rect = pygame.Rect(image_x - 2, image_y - 2,
                                IMAGE_DISPLAY_SIZE[0] + 4, IMAGE_DISPLAY_SIZE[1] + 4)
        pygame.draw.rect(screen, WHITE, border_rect)
        
        # Draw the image
        screen.blit(current_detection_image, (image_x, image_y))
        
        # Draw detection information
        draw_detection_info(screen, (image_x, image_y), current_detection_data)

def main():
    # Set up the file system observer
    observer = setup_motion_detection_observer()
    
    running = True
    clock = pygame.time.Clock()
    global show_detection_info
    
    try:
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Calculate room position
                        room_x = (WINDOW_WIDTH - ROOM_WIDTH) // 2
                        room_y = (WINDOW_HEIGHT - ROOM_HEIGHT) // 2
                        
                        # Get node position
                        data = load_nodes()
                        for node in data["nodes"]:
                            if node["id"] == 1:
                                node_pos = (room_x + node["coordinates"]["x"],
                                          room_y + node["coordinates"]["y"])
                                # Check if click is on node
                                if is_point_in_circle(event.pos, node_pos, NODE_RADIUS):
                                    show_detection_info = not show_detection_info
                                break
            
            # Draw everything
            draw_battlefield()
            
            # Update display
            pygame.display.flip()
            
            # Cap at 60 FPS
            clock.tick(60)
    
    finally:
        # Clean up
        observer.stop()
        observer.join()
        pygame.quit()

if __name__ == "__main__":
    main()