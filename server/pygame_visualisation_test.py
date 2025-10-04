import pygame as pg
from pathlib import Path
import json

# ------------- CONFIG -------------
WIDTH, HEIGHT = 1000, 700
BG_COLOR = (20, 22, 28)
EDGE_COLOR = (170, 170, 190)
NODE_COLOR = (240, 240, 255)
NODE_RADIUS = 10
FONT_NAME = pg.font.get_default_font()


NODES = {}
nodes_ping_status = {}

with open('server/nodes.json', 'r') as f:
    data = json.load(f)
    nodes = data['nodes']
    for node in nodes:
        name = str(node['id'])
        x = node['coordinates']['x']
        y = node['coordinates']['y']
        NODES[name] = (x, y)
        nodes_ping_status[name] = False  # Initially, all nodes are unpinged

nodes_ping_status['2'] = True


IMAGE_SIZE_WORLD = (48, 48)  # size in *world* units so it scales with zoom

# ------------- HELPERS -------------

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
                if event.button == 1:  # left click starts drag
                    dragging = True
                    last_mouse = pg.Vector2(event.pos)
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
            pg.draw.circle(screen, NODE_COLOR, p, r)
            label = font.render(node_id, True, (10, 10, 10))
            screen.blit(label, (p.x - label.get_width() // 2, p.y - label.get_height() // 2))
            if nodes_ping_status[node_id]:
                pg.draw.circle(screen, (255, 0, 0), p, r+4, 2)  # red ring for pinged nodes

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
