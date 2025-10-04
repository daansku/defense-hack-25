import pygame
import json
import os
import time
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, Optional, Tuple, List
from collections import deque

# Initialize Pygame
pygame.init()

# Initialize font
pygame.font.init()
font = pygame.font.SysFont('JetBrains Mono', 16)
node_id_font = pygame.font.SysFont('JetBrains Mono', 16)

# Window dimensions
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

# Constants
NODE_RADIUS = 8
FLASH_DURATION = 1.0  # Duration of flash in seconds
DETECTION_CHECK_INTERVAL = 1.0  # How often to check for detections (seconds)
IMAGE_DISPLAY_SIZE = (200, 150)  # Size for the detection image thumbnail
MAX_MESSAGES = 10  # Maximum number of messages to store
BUTTON_HEIGHT = 30
BUTTON_WIDTH = 120
BUTTON_PADDING = 10

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (100, 100, 100)

class DetectionMessage:
    def __init__(self, node_id: int, object_counts: Dict[str, int], timestamp: float):
        self.node_id = node_id
        self.object_counts = object_counts
        self.timestamp = timestamp
        
    def get_message(self) -> str:
        objects = []
        for obj_type, count in self.object_counts.items():
            if count == 1:
                objects.append(f"1 {obj_type}")
            else:
                objects.append(f"{count} {obj_type}s")
        object_text = ", ".join(objects)
        return f"Object detected at Node {self.node_id}: {object_text}"

# Node states
class NodeState:
    def __init__(self):
        self.is_flashing = False
        self.last_detection_time = 0
        self.active_detection = False
        self.current_detection_image = None
        self.current_image_path = None
        self.current_detection_data = None
        self.show_detection_info = False
        self.last_message_time = 0

# Button class for message toggle
class Button:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.is_messages_visible = True
        
    def draw(self, screen):
        # Draw button background
        color = LIGHT_GRAY if self.is_messages_visible else DARK_GRAY
        pygame.draw.rect(screen, color, self.rect)
        
        # Draw button border
        pygame.draw.rect(screen, WHITE, self.rect, 1)
        
        # Draw button text
        text = font.render("Hide Messages" if self.is_messages_visible else "Show Messages", True, WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
    
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.is_messages_visible = not self.is_messages_visible
            return True
        return False

# Global variables
node_states: Dict[int, NodeState] = {1: NodeState(), 2: NodeState(), 3: NodeState()}
last_detection_check = 0
detection_messages: deque = deque(maxlen=MAX_MESSAGES)
message_button = Button(
    BUTTON_PADDING,
    WINDOW_HEIGHT - BUTTON_HEIGHT - BUTTON_PADDING,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)

def load_floormap():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    floormap_path = os.path.join(os.path.dirname(script_dir), "floormap.png")
    try:
        original_map = pygame.image.load(floormap_path)
        # Calculate scaling factors for both dimensions
        scale_x = WINDOW_WIDTH / original_map.get_width()
        scale_y = WINDOW_HEIGHT / original_map.get_height()
        # Use the smaller scaling factor to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(original_map.get_width() * scale)
        new_height = int(original_map.get_height() * scale)
        
        # Scale the image
        scaled_map = pygame.transform.scale(original_map, (new_width, new_height))
        
        # Create a black surface the size of the window
        final_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        final_surface.fill((0, 0, 0))
        
        # Calculate position to center the image
        x_offset = (WINDOW_WIDTH - new_width) // 2
        y_offset = (WINDOW_HEIGHT - new_height) // 2
        
        # Blit the scaled image onto the surface
        final_surface.blit(scaled_map, (x_offset, y_offset))
        
        return {
            'surface': final_surface,
            'scale': scale,
            'offset': (x_offset, y_offset)
        }
    except (pygame.error, FileNotFoundError) as e:
        print(f"Error loading floormap: {e}")
        return None

# Load floormap and get scaling information
FLOORMAP_DATA = load_floormap()

# Create the window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Battlefield UI")

class MotionDetectionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            # We'll handle the flashing when we detect which node it's for
            pass

def setup_motion_detection_observer():
    """Set up the file system observer for the motion_detected directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    motion_detected_dir = os.path.join(os.path.dirname(script_dir), "motion_detected")
    
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
    """Check if there are any active detections for nodes"""
    global last_detection_check
    
    current_time = time.time()
    if current_time - last_detection_check < DETECTION_CHECK_INTERVAL:
        return
    
    last_detection_check = current_time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detections_dir = os.path.join(os.path.dirname(script_dir), "detections")
    
    if not os.path.exists(detections_dir):
        for state in node_states.values():
            state.active_detection = False
            state.current_detection_image = None
            state.current_image_path = None
            state.current_detection_data = None
        return
    
    # Get all detection files
    detection_files = glob.glob(os.path.join(detections_dir, "detection_*.json"))
    
    # Track latest detection for each node
    latest_detections = {node_id: {'time': 0, 'data': None} for node_id in node_states.keys()}
    
    # Process all detection files
    for detection_file in detection_files:
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
                node_id = data.get("node_id")
                if node_id in node_states:
                    timestamp = time.mktime(time.strptime(data["timestamp"].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
                    if timestamp > latest_detections[node_id]['time']:
                        latest_detections[node_id]['time'] = timestamp
                        latest_detections[node_id]['data'] = data
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    
    current_time = time.time()
    
    # Update node states with latest detections
    for node_id, latest in latest_detections.items():
        state = node_states[node_id]
        if latest['data']:
            # Check if this is a new detection we haven't messaged about
            if latest['time'] > state.last_message_time:
                # Add new detection message
                detection_messages.append(DetectionMessage(
                    node_id=node_id,
                    object_counts=latest['data']["object_counts"],
                    timestamp=current_time
                ))
                state.last_message_time = latest['time']
            
            state.active_detection = True
            state.current_detection_data = latest['data']
            # Load the associated image if it's different from the current one
            image_path = os.path.join(os.path.dirname(script_dir), 
                                    latest['data']["image_path"].replace('\\', '/'))
            if image_path != state.current_image_path:
                state.current_image_path = image_path
                state.current_detection_image = load_and_scale_image(image_path)
                state.last_detection_time = current_time
                state.is_flashing = True
        else:
            state.active_detection = False
            state.current_detection_image = None
            state.current_image_path = None
            state.current_detection_data = None

def load_nodes():
    """Load nodes from the JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "nodes.json")
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def get_node_color(node_id: int) -> Tuple[int, int, int]:
    """Determine the node color based on flash state and active detections"""
    state = node_states[node_id]
    
    if state.is_flashing or state.active_detection:
        current_time = time.time()
        if state.is_flashing and current_time - state.last_detection_time > FLASH_DURATION:
            state.is_flashing = False
            if not state.active_detection:
                return WHITE
        
        # Flash effect: alternate between red and white every 0.2 seconds
        return RED if int(current_time * 5) % 2 == 0 else WHITE
    
    return WHITE

def scale_coordinates(x: int, y: int) -> Tuple[int, int]:
    """Scale coordinates based on the floormap scaling"""
    if FLOORMAP_DATA:
        scale = FLOORMAP_DATA['scale']
        offset_x, offset_y = FLOORMAP_DATA['offset']
        return (int(x * scale) + offset_x, int(y * scale) + offset_y)
    return (x, y)

def is_point_in_circle(point: Tuple[int, int], center: Tuple[int, int], radius: int) -> bool:
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

def draw_node_id(screen, node_pos: Tuple[int, int], node_id: int):
    """Draw node ID in the bottom right of the node"""
    # Create text surface for node ID
    id_text = node_id_font.render(str(node_id), True, WHITE)
    text_rect = id_text.get_rect()
    
    # Position text at bottom right of node
    text_pos = (node_pos[0] + NODE_RADIUS, node_pos[1] + NODE_RADIUS)
    screen.blit(id_text, text_pos)

def draw_detection_messages(screen):
    """Draw detection messages in bottom left corner"""
    if not message_button.is_messages_visible:
        message_button.draw(screen)
        return
        
    # Draw all messages
    y_offset = WINDOW_HEIGHT - BUTTON_HEIGHT - BUTTON_PADDING - 10  # Start above the button
    for message in reversed(detection_messages):  # Show newest messages at the bottom
        text = font.render(message.get_message(), True, RED)
        text_rect = text.get_rect()
        text_rect.bottomleft = (10, y_offset)
        screen.blit(text, text_rect)
        y_offset -= 25  # Move up for next message
    
    # Draw the toggle button
    message_button.draw(screen)

def draw_battlefield():
    """Draw the battlefield with room and nodes"""
    # Fill background with black
    screen.fill(BLACK)
    
    # Draw the floormap as background
    if FLOORMAP_DATA:
        screen.blit(FLOORMAP_DATA['surface'], (0, 0))
    
    # Check for active detections
    check_active_detections()
    
    # Load and draw nodes
    data = load_nodes()
    
    # Draw all monitored nodes
    for node in data["nodes"]:
        if node["id"] in node_states:
            # Scale the node coordinates
            node_pos = scale_coordinates(node["coordinates"]["x"], node["coordinates"]["y"])
            
            # Draw the node with dynamic color
            node_color = get_node_color(node["id"])
            pygame.draw.circle(screen, node_color, node_pos, NODE_RADIUS)
            
            # Draw node ID
            draw_node_id(screen, node_pos, node["id"])
            
            # Draw detection info if toggled on for this node
            state = node_states[node["id"]]
            if state.show_detection_info and state.current_detection_image:
                # Position the image to the right of the node
                image_x = node_pos[0] + NODE_RADIUS + 10
                image_y = node_pos[1] - IMAGE_DISPLAY_SIZE[1] // 2
                
                # Draw a white border around the image
                border_rect = pygame.Rect(image_x - 2, image_y - 2,
                                        IMAGE_DISPLAY_SIZE[0] + 4, IMAGE_DISPLAY_SIZE[1] + 4)
                pygame.draw.rect(screen, WHITE, border_rect)
                
                # Draw the image
                screen.blit(state.current_detection_image, (image_x, image_y))
                
                # Draw detection information
                draw_detection_info(screen, (image_x, image_y), state.current_detection_data)
    
    # Draw detection messages
    draw_detection_messages(screen)

def main():
    # Set up the file system observer
    observer = setup_motion_detection_observer()
    
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check if message button was clicked
                        if message_button.handle_click(event.pos):
                            continue
                            
                        # Check all monitored nodes for clicks
                        data = load_nodes()
                        for node in data["nodes"]:
                            if node["id"] in node_states:
                                node_pos = scale_coordinates(
                                    node["coordinates"]["x"],
                                    node["coordinates"]["y"]
                                )
                                if is_point_in_circle(event.pos, node_pos, NODE_RADIUS):
                                    node_states[node["id"]].show_detection_info = not node_states[node["id"]].show_detection_info
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