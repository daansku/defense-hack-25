import pygame as pg
from pathlib import Path
import json
import numpy as np

# ------------- CONFIG -------------
WIDTH, HEIGHT = 1000, 700
BG_COLOR = (20, 22, 28)
EDGE_COLOR = (170, 170, 190)
NODE_COLOR = (240, 240, 255)
NODE_RADIUS = 10
FONT_NAME = pg.font.get_default_font()


NODES = {}
nodes_ping_status = {}
clicked_nodes = {}
location_names = {}
FRIENDLY_POSITIONS = {}  # Store friendly position coordinates
friendly_names = {}  # Store friendly position names

with open('server/nodes.json', 'r') as f:
    data = json.load(f)
    
    # Load regular nodes
    nodes = data['nodes']
    for node in nodes:
        name = str(node['id'])
        x = node['coordinates']['x']
        y = node['coordinates']['y']
        NODES[name] = (x, y)
        nodes_ping_status[name] = False  # Initially, all nodes are unpinged
        clicked_nodes[name] = False  # Initially, no nodes are clicked
        location_names[name] = node['location_name']  # Store the location name
    
    # Load friendly positions
    friendly_positions = data['friendly_positions']
    for pos in friendly_positions:
        name = str(pos['id'])
        x = pos['coordinates']['x']
        y = pos['coordinates']['y']
        FRIENDLY_POSITIONS[name] = (x, y)
        friendly_names[name] = pos['name']  # Store the friendly position name

nodes_ping_status['2'] = True


IMAGE_SIZE_WORLD = (48, 48)  # size in *world* units so it scales with zoom

# ------------- HELPERS -------------

def calculate_regression_line(points):
    """Calculate regression line parameters for a set of points."""
    if len(points) < 2:
        return None
    
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # Calculate regression line
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Get line endpoints for visualization
    x_min, x_max = min(x), max(x)
    y_min = m * x_min + c
    y_max = m * x_max + c
    
    return (x_min, y_min), (x_max, y_max)

def load_image_or_placeholder(path_str: str, size: tuple[int, int]) -> pg.Surface:
    """Try loading an image; if missing, return a simple placeholder surface."""
    path = Path(path_str)
    if path.exists():
        img = pg.image.load(str(path)).convert_alpha()
        return pg.transform.smoothscale(img, size)
    # placeholder
    surf = pg.Surface(size, pg.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pg.draw.rect(surf, (60, 180, 255), surf.get_rect(), border_radius=8)
    pg.draw.line(surf, (255, 255, 255), (0, 0), (size[0], size[1]), 2)
    pg.draw.line(surf, (255, 255, 255), (0, size[1]), (size[0], 0), 2)
    return surf

class Camera:
    """Simple pan/zoom camera mapping world -> screen coordinates."""
    def __init__(self, pos=(0, 0), zoom=1.0):
        self.offset = pg.Vector2(pos)  # world-space top-left visible
        self.zoom = zoom

    def world_to_screen(self, world_pos):
        w = pg.Vector2(world_pos)
        return (w - self.offset) * self.zoom

    def screen_to_world(self, screen_pos):
        s = pg.Vector2(screen_pos)
        return s / self.zoom + self.offset

    def pan(self, delta_screen):
        # convert screen delta to world delta so pan speed feels consistent
        self.offset -= pg.Vector2(delta_screen) / self.zoom

# ------------- MAIN ---------------

def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Graph Map with Images — Pygame")
    clock = pg.time.Clock()
    font = pg.font.SysFont(FONT_NAME, 16)

    # Pre-load images at a neutral size (we’ll rescale per-zoom each frame)
    # base_images = {
    #     node_id: load_image_or_placeholder(path, IMAGE_SIZE_WORLD)
    #     for node_id, path in NODE_IMAGES.items()
    # }

    camera = Camera(pos=(-100, -60), zoom=1.5)  # initial view
    dragging = False
    last_mouse = pg.Vector2(0, 0)

    running = True
    while running:
        dt = clock.tick(60) / 1000  # seconds
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = pg.Vector2(event.pos)
                if event.button == 1:  # left click
                    # Check if clicked on any node
                    world_mouse = camera.screen_to_world(mouse_pos)
                    for node_id, pos in NODES.items():
                        node_screen_pos = camera.world_to_screen(pos)
                        distance = (node_screen_pos - mouse_pos).length()
                        if distance <= NODE_RADIUS * camera.zoom:
                            clicked_nodes[node_id] = not clicked_nodes[node_id]  # Toggle node state
                            break
                    # If not clicked on a node, start dragging
                    dragging = True
                    last_mouse = mouse_pos
                elif event.button == 4:  # wheel up = zoom in
                    zoom_before = camera.zoom
                    camera.zoom = min(5.0, camera.zoom * 1.1)
                    # zoom to cursor: adjust offset so cursor stays on same world point
                    mouse = pg.Vector2(event.pos)
                    world_under = camera.screen_to_world(mouse)
                    camera.offset = world_under - (mouse / camera.zoom)
                elif event.button == 5:  # wheel down = zoom out
                    zoom_before = camera.zoom
                    camera.zoom = max(0.2, camera.zoom / 1.1)
                    mouse = pg.Vector2(event.pos)
                    world_under = camera.screen_to_world(mouse)
                    camera.offset = world_under - (mouse / camera.zoom)

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pg.MOUSEMOTION and dragging:
                camera.pan(pg.Vector2(event.pos) - last_mouse)
                last_mouse = pg.Vector2(event.pos)

        # --------- DRAW ----------
        screen.fill(BG_COLOR)

        # Draw edges
        # for a, b in EDGES:
        #     pa = camera.world_to_screen(NODES[a])
        #     pb = camera.world_to_screen(NODES[b])
        #     pg.draw.line(screen, EDGE_COLOR, pa, pb, 2)

        # Draw nodes (circles) + labels
        for node_id, pos in NODES.items():
            
            p = camera.world_to_screen(pos)
            r = max(2, int(NODE_RADIUS * camera.zoom * 0.8))
            # Draw node with red color if clicked, otherwise use default color
            node_color = (255, 0, 0) if clicked_nodes[node_id] else NODE_COLOR
            pg.draw.circle(screen, node_color, p, r)
            
            # Draw location name
            location = location_names[node_id]
            label = font.render(location, True, (240, 240, 255))  # White text for better visibility
            
            # Draw a semi-transparent background for the text
            label_bg = pg.Surface((label.get_width() + 10, label.get_height() + 6))
            label_bg.fill((40, 42, 48))  # Darker background color
            label_bg.set_alpha(200)  # Semi-transparent
            screen.blit(label_bg, (p.x - label_bg.get_width() // 2, p.y + r + 5))
            
            # Draw the text
            screen.blit(label, (p.x - label.get_width() // 2, p.y + r + 8))
            
       
        # Get clicked nodes and draw regression line if more than one
        clicked_points = []
        for node_id, is_clicked in clicked_nodes.items():
            if is_clicked:
                clicked_points.append(NODES[node_id])
        
        # Draw regression line if we have multiple clicked points
        if len(clicked_points) >= 2:
            # Calculate regression line in world coordinates
            line_points = calculate_regression_line(clicked_points)
            if line_points:
                start_point, end_point = line_points
                # Convert to screen coordinates
                start_screen = camera.world_to_screen(start_point)
                end_screen = camera.world_to_screen(end_point)
                # Draw the line 
            pg.draw.line(screen, (255, 100, 100), start_screen, end_screen, 4)

        # Draw friendly positions (in green)
        for pos_id, pos in FRIENDLY_POSITIONS.items():
            p = camera.world_to_screen(pos)
            r = max(2, int(NODE_RADIUS * camera.zoom * 0.8))
            
            # Draw green node
            pg.draw.circle(screen, (0, 255, 0), p, r)
            
            # Draw location name
            location = friendly_names[pos_id]
            label = font.render(location, True, (200, 255, 200))  # Light green text
            
            # Draw a semi-transparent background for the text
            label_bg = pg.Surface((label.get_width() + 10, label.get_height() + 6))
            label_bg.fill((40, 48, 40))  # Dark green background
            label_bg.set_alpha(200)  # Semi-transparent
            screen.blit(label_bg, (p.x - label_bg.get_width() // 2, p.y + r + 5))
            
            # Draw the text
            screen.blit(label, (p.x - label.get_width() // 2, p.y + r + 8))

        # # Draw images (scaled with zoom)
        # for node_id, base_img in base_images.items():
        #     world_xy = pg.Vector2(NODES[node_id])
        #     screen_xy = camera.world_to_screen(world_xy)
        #     # scale image by zoom
        #     w, h = base_img.get_size()
        #     scaled = pg.transform.smoothscale(base_img, (max(1, int(w * camera.zoom)), max(1, int(h * camera.zoom))))
        #     rect = scaled.get_rect(center=(screen_xy.x, screen_xy.y - 26 * camera.zoom))  # float above node a bit
        #     screen.blit(scaled, rect)

        # HUD
        # hud = font.render(f"Zoom: {camera.zoom:.2f}  |  Drag: LMB  |  Wheel: Zoom  |  ESC to quit", True, (200, 200, 210))
        # screen.blit(hud, (12, HEIGHT - 24))

        pg.display.flip()

        # ESC to quit
        keys = pg.key.get_pressed()
        if keys[pg.K_ESCAPE]:
            running = False

    pg.quit()

if __name__ == "__main__":
    main()
