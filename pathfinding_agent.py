"""
Dynamic Pathfinding Agent
=========================
Implements GBFS and A* with Manhattan/Euclidean heuristics.
Dynamic obstacles, real-time re-planning, Pygame GUI.

Requirements:
    pip install pygame

Run:
    python pathfinding_agent.py
"""

import pygame
import sys
import random
import math
import heapq
import time
from tkinter import simpledialog, messagebox
import tkinter as tk

# ─────────────────────────── CONSTANTS ──────────────────────────────────────

# Colours
WHITE       = (255, 255, 255)
BLACK       = (20,  20,  20)
GRAY        = (180, 180, 180)
DARK_GRAY   = (100, 100, 100)
RED         = (220,  60,  60)   # visited / explored
YELLOW      = (255, 220,  50)   # frontier
GREEN       = ( 50, 200,  80)   # final path
BLUE        = ( 50, 120, 220)   # start
ORANGE      = (255, 140,   0)   # goal
WALL_COLOR  = ( 40,  40,  40)
BG_COLOR    = ( 30,  30,  30)
PANEL_COLOR = ( 45,  45,  55)
TEXT_COLOR  = (230, 230, 230)
BTN_COLOR   = ( 70,  70,  90)
BTN_HOVER   = ( 90,  90, 120)
BTN_ACTIVE  = ( 50, 180, 120)
AGENT_COLOR = (255,  80, 180)

# Layout
PANEL_W     = 260
MIN_CELL    = 8
MAX_CELL    = 60
FPS         = 30

# ─────────────────────────── GRID ───────────────────────────────────────────

class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.walls = set()
        self.start = (0, 0)
        self.goal  = (rows - 1, cols - 1)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r, c):
        return self.in_bounds(r, c) and (r, c) not in self.walls

    def neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.is_walkable(nr, nc):
                yield (nr, nc)

    def generate_random(self, density=0.30):
        self.walls.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.goal):
                    continue
                if random.random() < density:
                    self.walls.add((r, c))

    def is_solvable(self):
        """BFS check."""
        sr, sc = self.start
        gr, gc = self.goal
        if not self.is_walkable(sr, sc) or not self.is_walkable(gr, gc):
            return False
        visited = {self.start}
        queue = [self.start]
        while queue:
            r, c = queue.pop(0)
            if (r, c) == self.goal:
                return True
            for nb in self.neighbors(r, c):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False


# ─────────────────────────── HEURISTICS ─────────────────────────────────────

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ─────────────────────────── SEARCH ALGORITHMS ──────────────────────────────

def search(grid, algorithm, heuristic_fn):
    """
    Returns (path, visited_order, frontier_snapshots, exec_ms)
    path: list of (r,c) from start to goal, or None
    visited_order: list of nodes in expansion order
    exec_ms: float
    """
    start = grid.start
    goal  = grid.goal
    t0 = time.perf_counter()

    # priority queue entries: (priority, counter, node)
    counter   = 0
    open_heap = []
    g_cost    = {start: 0}
    came_from = {start: None}
    visited   = []       # expansion order
    in_open   = {start}

    h0 = heuristic_fn(start, goal)
    f0 = h0 if algorithm == 'GBFS' else (0 + h0)
    heapq.heappush(open_heap, (f0, counter, start))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        in_open.discard(current)

        if current == goal:
            # reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            exec_ms = (time.perf_counter() - t0) * 1000
            return path, visited, exec_ms

        visited.append(current)

        for nb in grid.neighbors(*current):
            new_g = g_cost[current] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb]    = new_g
                came_from[nb] = current
                h = heuristic_fn(nb, goal)
                f = h if algorithm == 'GBFS' else (new_g + h)
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))
                in_open.add(nb)

    exec_ms = (time.perf_counter() - t0) * 1000
    return None, visited, exec_ms


# ─────────────────────────── BUTTON ──────────────────────────────────────────

class Button:
    def __init__(self, rect, label, toggle=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.toggle = toggle
        self.active = False
        self.font   = None

    def draw(self, surf):
        if self.font is None:
            self.font = pygame.font.SysFont('segoeui', 10, bold=True)
        mx, my = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mx, my)
        if self.active:
            colour = BTN_ACTIVE
        elif hovered:
            colour = BTN_HOVER
        else:
            colour = BTN_COLOR
        pygame.draw.rect(surf, colour, self.rect, border_radius=6)
        pygame.draw.rect(surf, GRAY, self.rect, 1, border_radius=6)
        txt = self.font.render(self.label, True, TEXT_COLOR)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        if self.rect.collidepoint(pos):
            if self.toggle:
                self.active = not self.active
            return True
        return False


# ─────────────────────────── APP ─────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        self.screen_w = 1100
        self.screen_h = 720
        self.screen   = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont('segoeui', 13)
        self.font_m = pygame.font.SysFont('segoeui', 15, bold=True)
        self.font_l = pygame.font.SysFont('segoeui', 18, bold=True)

        # Grid config
        self.rows = 20
        self.cols = 30
        self.grid = Grid(self.rows, self.cols)
        self.grid.generate_random(0.25)
        while not self.grid.is_solvable():
            self.grid.generate_random(0.25)

        # State
        self.algorithm    = 'A*'        # 'GBFS' or 'A*'
        self.heuristic    = 'Manhattan' # 'Manhattan' or 'Euclidean'
        self.mode         = 'IDLE'      # IDLE, SEARCHING, ANIMATING, DONE, DYNAMIC
        self.edit_mode    = 'WALL'      # WALL, START, GOAL

        self.path         = None
        self.visited      = []
        self.frontier_set = set()
        self.anim_idx     = 0
        self.anim_visited = []
        self.anim_path    = []
        self.agent_pos    = None        # for dynamic mode
        self.agent_step   = 0
        self.dyn_prob     = 0.04        # probability a new wall spawns per step

        # Metrics
        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_ms       = 0.0
        self.status_msg    = "Ready. Generate map or draw walls, then press Search."
        self.replanning    = False

        # Display sets
        self.vis_visited  = set()
        self.vis_frontier = set()
        self.vis_path     = set()

        self._build_buttons()

    # ── UI LAYOUT ─────────────────────────────────────────────────────────

    def _build_buttons(self):
        x  = self.screen_w - PANEL_W + 10
        bw = PANEL_W - 20
        bh = 30

        def B(y, label, toggle=False):
            return Button((x, y, bw, bh), label, toggle)

        self.btn_gen      = B(40,  "Generate Random Map")
        self.btn_clear    = B(70, "Clear All Walls")
        self.btn_search   = B(100, "▶  Search")
        self.btn_step     = B(130, "Step-by-Step")
        self.btn_reset    = B(160, "Reset Visualisation")

        self.btn_algo_astar = B(190, "Algorithm: A*",    toggle=True)
        self.btn_algo_gbfs  = B(220, "Algorithm: GBFS",  toggle=True)
        self.btn_algo_astar.active = True

        self.btn_heur_man  = B(250, "Heuristic: Manhattan", toggle=True)
        self.btn_heur_euc  = B(280, "Heuristic: Euclidean", toggle=True)
        self.btn_heur_man.active = True

        self.btn_edit_wall  = B(310, "Edit: Place Wall",  toggle=True)
        self.btn_edit_start = B(340, "Edit: Move Start",  toggle=True)
        self.btn_edit_goal  = B(370, "Edit: Move Goal",   toggle=True)
        self.btn_edit_wall.active = True

        self.btn_dynamic    = B(400, "Dynamic Mode",      toggle=True)
        self.btn_set_size   = B(430, "Set Grid Size")
        self.btn_set_density= B(460, "Set Obstacle Density")

        self.all_buttons = [
            self.btn_gen, self.btn_clear, self.btn_search, self.btn_step,
            self.btn_reset, self.btn_algo_astar, self.btn_algo_gbfs,
            self.btn_heur_man, self.btn_heur_euc,
            self.btn_edit_wall, self.btn_edit_start, self.btn_edit_goal,
            self.btn_dynamic, self.btn_set_size, self.btn_set_density,
        ]

    def _reposition_buttons(self):
        """Recompute button x positions after resize."""
        x  = self.screen_w - PANEL_W + 10
        bw = PANEL_W - 20
        for btn in self.all_buttons:
            btn.rect.x = x
            btn.rect.width = bw

    # ── CELL GEOMETRY ─────────────────────────────────────────────────────

    @property
    def grid_area_w(self):
        return self.screen_w - PANEL_W

    @property
    def cell_size(self):
        cw = self.grid_area_w // self.cols
        ch = self.screen_h // self.rows
        return max(MIN_CELL, min(MAX_CELL, min(cw, ch)))

    @property
    def offset_x(self):
        return (self.grid_area_w - self.cols * self.cell_size) // 2

    @property
    def offset_y(self):
        return (self.screen_h - self.rows * self.cell_size) // 2

    def cell_rect(self, r, c):
        cs = self.cell_size
        x  = self.offset_x + c * cs
        y  = self.offset_y + r * cs
        return pygame.Rect(x, y, cs, cs)

    def pixel_to_cell(self, px, py):
        cs = self.cell_size
        c  = (px - self.offset_x) // cs
        r  = (py - self.offset_y) // cs
        if self.grid.in_bounds(r, c):
            return (r, c)
        return None

    # ── SEARCH HELPERS ────────────────────────────────────────────────────

    def _heuristic_fn(self):
        return manhattan if self.heuristic == 'Manhattan' else euclidean

    def run_search(self, start_override=None):
        g = self.grid
        if start_override:
            old_start   = g.start
            g.start     = start_override
        path, visited, exec_ms = search(g, self.algorithm, self._heuristic_fn())
        if start_override:
            g.start = old_start

        self.visited  = visited
        self.exec_ms  = exec_ms
        self.path     = path
        self.nodes_visited = len(visited)
        self.path_cost     = (len(path) - 1) if path else 0
        return path, visited

    def start_full_search(self):
        self.vis_visited.clear()
        self.vis_frontier.clear()
        self.vis_path.clear()
        self.anim_idx    = 0
        path, visited    = self.run_search()
        self.anim_visited = visited[:]
        self.anim_path    = path[:] if path else []
        # Show everything instantly
        self.vis_visited  = set(visited)
        self.vis_path     = set(path) if path else set()
        self.mode         = 'DONE'
        if not path:
            self.status_msg = "No path found!"
        else:
            self.status_msg = (f"Done! Nodes: {self.nodes_visited} | "
                               f"Cost: {self.path_cost} | "
                               f"Time: {self.exec_ms:.2f}ms")

    def start_step_search(self):
        self.vis_visited.clear()
        self.vis_frontier.clear()
        self.vis_path.clear()
        path, visited    = self.run_search()
        self.anim_visited = visited[:]
        self.anim_path    = path[:] if path else []
        self.anim_idx     = 0
        self.mode         = 'ANIMATING'
        self.status_msg   = "Animating... (searching)"

    def reset_vis(self):
        self.vis_visited.clear()
        self.vis_frontier.clear()
        self.vis_path.clear()
        self.anim_idx   = 0
        self.anim_visited = []
        self.anim_path    = []
        self.agent_pos    = None
        self.agent_step   = 0
        self.mode         = 'IDLE'
        self.status_msg   = "Visualisation reset."

    # ── DYNAMIC MODE ──────────────────────────────────────────────────────

    def dynamic_tick(self):
        """Called every frame when dynamic mode is running."""
        if self.path is None or self.agent_step >= len(self.path):
            self.mode = 'DONE'
            self.status_msg = "Agent reached goal!" if self.path else "No path!"
            return

        self.agent_pos = self.path[self.agent_step]
        self.agent_step += 1

        # Spawn new obstacle with small probability
        if random.random() < self.dyn_prob:
            # pick a random walkable cell that's not start/goal/agent/path
            candidates = []
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    node = (r, c)
                    if (node not in self.grid.walls and
                            node != self.grid.start and
                            node != self.grid.goal and
                            node != self.agent_pos):
                        candidates.append(node)
            if candidates:
                new_wall = random.choice(candidates)
                self.grid.walls.add(new_wall)
                # Check if new wall is on current remaining path
                remaining = self.path[self.agent_step:]
                if new_wall in remaining:
                    # Re-plan from current position
                    self.status_msg = f"Obstacle at {new_wall}! Re-planning..."
                    self.replanning = True
                    old_start = self.grid.start
                    self.grid.start = self.agent_pos
                    new_path, new_visited, new_ms = search(
                        self.grid, self.algorithm, self._heuristic_fn())
                    self.grid.start = old_start

                    self.exec_ms      += new_ms
                    self.nodes_visited += len(new_visited)
                    if new_path:
                        self.path       = new_path
                        self.agent_step = 0
                        self.vis_visited.update(new_visited)
                        self.vis_path    = set(new_path)
                        self.path_cost   = len(new_path) - 1
                        self.status_msg  = (f"Re-planned! Cost:{self.path_cost} "
                                            f"Nodes:{self.nodes_visited} "
                                            f"Time:{self.exec_ms:.1f}ms")
                    else:
                        self.path = None
                        self.mode = 'DONE'
                        self.status_msg = "Re-plan failed – no path!"
                    self.replanning = False

    # ── DRAWING ───────────────────────────────────────────────────────────

    def draw_grid(self):
        cs  = self.cell_size
        ox  = self.offset_x
        oy  = self.offset_y

        for r in range(self.rows):
            for c in range(self.cols):
                node = (r, c)
                rect = self.cell_rect(r, c)

                if node in self.grid.walls:
                    colour = WALL_COLOR
                elif node == self.grid.start:
                    colour = BLUE
                elif node == self.grid.goal:
                    colour = ORANGE
                elif self.agent_pos and node == self.agent_pos:
                    colour = AGENT_COLOR
                elif node in self.vis_path:
                    colour = GREEN
                elif node in self.vis_visited:
                    colour = RED
                elif node in self.vis_frontier:
                    colour = YELLOW
                else:
                    colour = WHITE

                pygame.draw.rect(self.screen, colour, rect)
                # grid lines
                if cs > 10:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)

        # Draw start/goal labels if large enough
        if cs >= 18:
            for label, node, col in [
                ("S", self.grid.start, WHITE),
                ("G", self.grid.goal,  WHITE),
            ]:
                rect = self.cell_rect(*node)
                txt  = self.font_m.render(label, True, col)
                self.screen.blit(txt, txt.get_rect(center=rect.center))

        # Agent marker
        if self.agent_pos and cs >= 10:
            rect = self.cell_rect(*self.agent_pos)
            pygame.draw.circle(self.screen, AGENT_COLOR,
                               rect.center, cs // 3)

    def draw_panel(self):
        px = self.screen_w - PANEL_W
        panel_rect = pygame.Rect(px, 0, PANEL_W, self.screen_h)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)
        pygame.draw.line(self.screen, GRAY, (px, 0), (px, self.screen_h), 2)

        # Title
        title = self.font_l.render("Pathfinding Agent", True, TEXT_COLOR)
        self.screen.blit(title, (px + 10, 10))

        # Buttons
        for btn in self.all_buttons:
            btn.draw(self.screen)

        # Metrics
        my = self.screen_h - 150
        pygame.draw.line(self.screen, DARK_GRAY, (px+20, my-5), (self.screen_w-10, my-5))
        metrics = [
            ("Algorithm",    self.algorithm),
            ("Heuristic",    self.heuristic),
            ("Nodes Visited", str(self.nodes_visited)),
            ("Path Cost",    str(self.path_cost)),
            ("Exec Time",    f"{self.exec_ms:.2f} ms"),
            ("Grid",         f"{self.rows}×{self.cols}"),
        ]
        for i, (k, v) in enumerate(metrics):
            ky = my + i * 22
            kt = self.font_s.render(k + ":", True, GRAY)
            vt = self.font_s.render(v, True, TEXT_COLOR)
            self.screen.blit(kt, (px+12, ky))
            self.screen.blit(vt, (px + 140, ky))

        # Status bar
        status_y = self.screen_h - 24
        pygame.draw.rect(self.screen, BG_COLOR,
                         pygame.Rect(0, status_y, self.screen_w, 24))
        st = self.font_s.render(self.status_msg, True, TEXT_COLOR)
        self.screen.blit(st, (8, status_y + 4))

        # Legend
        legend = [
            (YELLOW,      "Frontier"),
            (RED,         "Visited"),
            (GREEN,       "Path"),
            (BLUE,        "Start"),
            (ORANGE,      "Goal"),
            (WALL_COLOR,  "Wall"),
            (AGENT_COLOR, "Agent"),
        ]
        lx = 10
        ly = 10
        for col, lbl in legend:
            pygame.draw.rect(self.screen, col,
                             pygame.Rect(lx, ly, 14, 14), border_radius=3)
            lt = self.font_s.render(lbl, True, TEXT_COLOR)
            self.screen.blit(lt, (lx+18, ly))
            lx += 120

    # ── ANIMATION TICK ────────────────────────────────────────────────────

    def animation_tick(self):
        steps_per_frame = max(1, len(self.anim_visited) // 120)
        for _ in range(steps_per_frame):
            if self.anim_idx < len(self.anim_visited):
                self.vis_visited.add(self.anim_visited[self.anim_idx])
                self.anim_idx += 1
            else:
                # Reveal path
                self.vis_path = set(self.anim_path) if self.anim_path else set()
                self.mode     = 'DONE'
                if self.anim_path:
                    self.status_msg = (f"Done! Nodes: {self.nodes_visited} | "
                                       f"Cost: {self.path_cost} | "
                                       f"Time: {self.exec_ms:.2f}ms")
                else:
                    self.status_msg = "No path found!"
                return

    # ── INPUT ─────────────────────────────────────────────────────────────

    def handle_click(self, pos, button):
        # Panel click?
        if pos[0] >= self.screen_w - PANEL_W:
            self._handle_panel_click(pos)
            return
        # Grid click
        cell = self.pixel_to_cell(*pos)
        if cell is None:
            return
        r, c = cell
        if button == 1:
            if self.edit_mode == 'WALL':
                if cell != self.grid.start and cell != self.grid.goal:
                    if cell in self.grid.walls:
                        self.grid.walls.discard(cell)
                    else:
                        self.grid.walls.add(cell)
                    # If we're done and a wall changed, reset vis
                    if self.mode == 'DONE':
                        self.reset_vis()
            elif self.edit_mode == 'START':
                if cell not in self.grid.walls and cell != self.grid.goal:
                    self.grid.start = cell
                    self.reset_vis()
            elif self.edit_mode == 'GOAL':
                if cell not in self.grid.walls and cell != self.grid.start:
                    self.grid.goal = cell
                    self.reset_vis()
        elif button == 3:
            # Right click always removes wall
            self.grid.walls.discard(cell)

    def _handle_panel_click(self, pos):
        if self.btn_gen.clicked(pos):
            self._ask_generate()
        elif self.btn_clear.clicked(pos):
            self.grid.walls.clear()
            self.reset_vis()
        elif self.btn_search.clicked(pos):
            self.start_full_search()
        elif self.btn_step.clicked(pos):
            self.start_step_search()
        elif self.btn_reset.clicked(pos):
            self.reset_vis()
        elif self.btn_algo_astar.clicked(pos):
            self.algorithm = 'A*'
            self.btn_algo_astar.active = True
            self.btn_algo_gbfs.active  = False
        elif self.btn_algo_gbfs.clicked(pos):
            self.algorithm = 'GBFS'
            self.btn_algo_gbfs.active  = True
            self.btn_algo_astar.active = False
        elif self.btn_heur_man.clicked(pos):
            self.heuristic = 'Manhattan'
            self.btn_heur_man.active = True
            self.btn_heur_euc.active = False
        elif self.btn_heur_euc.clicked(pos):
            self.heuristic = 'Euclidean'
            self.btn_heur_euc.active = True
            self.btn_heur_man.active = False
        elif self.btn_edit_wall.clicked(pos):
            self.edit_mode = 'WALL'
            self.btn_edit_wall.active  = True
            self.btn_edit_start.active = False
            self.btn_edit_goal.active  = False
        elif self.btn_edit_start.clicked(pos):
            self.edit_mode = 'START'
            self.btn_edit_start.active = True
            self.btn_edit_wall.active  = False
            self.btn_edit_goal.active  = False
        elif self.btn_edit_goal.clicked(pos):
            self.edit_mode = 'GOAL'
            self.btn_edit_goal.active  = True
            self.btn_edit_wall.active  = False
            self.btn_edit_start.active = False
        elif self.btn_dynamic.clicked(pos):
            if self.btn_dynamic.active:
                self._launch_dynamic()
            else:
                self.mode = 'IDLE'
                self.agent_pos  = None
                self.agent_step = 0
                self.status_msg = "Dynamic mode disabled."
        elif self.btn_set_size.clicked(pos):
            self._ask_grid_size()
        elif self.btn_set_density.clicked(pos):
            self._ask_density()

    def _ask_generate(self):
        root = tk.Tk(); root.withdraw()
        d = simpledialog.askfloat("Obstacle Density",
                                  "Enter density (0.0 – 0.7):",
                                  minvalue=0.0, maxvalue=0.7,
                                  initialvalue=0.28)
        root.destroy()
        if d is None:
            return
        attempts = 0
        while attempts < 200:
            self.grid.generate_random(d)
            if self.grid.is_solvable():
                break
            attempts += 1
        self.reset_vis()
        self.status_msg = f"Map generated (density={d:.2f})."

    def _ask_grid_size(self):
        root = tk.Tk(); root.withdraw()
        r = simpledialog.askinteger("Rows", "Number of rows (5-60):",
                                    minvalue=5, maxvalue=60, initialvalue=self.rows)
        c = simpledialog.askinteger("Cols", "Number of columns (5-80):",
                                    minvalue=5, maxvalue=80, initialvalue=self.cols)
        root.destroy()
        if r and c:
            self.rows = r
            self.cols = c
            self.grid = Grid(r, c)
            self.grid.generate_random(0.25)
            while not self.grid.is_solvable():
                self.grid.generate_random(0.25)
            self.reset_vis()
            self.status_msg = f"Grid resized to {r}×{c}."

    def _ask_density(self):
        root = tk.Tk(); root.withdraw()
        d = simpledialog.askfloat("Obstacle Density",
                                  "Density for next generation (0.0–0.7):",
                                  minvalue=0.0, maxvalue=0.7, initialvalue=0.28)
        root.destroy()
        if d is not None:
            self.dyn_prob = 0.02 + d * 0.08
            self.status_msg = f"Dynamic spawn prob set to {self.dyn_prob:.3f}."

    def _launch_dynamic(self):
        # First run search, then animate agent along path with dynamic obstacles
        path, visited = self.run_search()
        if not path:
            self.status_msg = "No path – cannot start dynamic mode."
            self.btn_dynamic.active = False
            return
        self.path       = path
        self.vis_path   = set(path)
        self.vis_visited= set(visited)
        self.agent_step = 0
        self.agent_pos  = path[0]
        self.mode       = 'DYNAMIC'
        self.status_msg = "Dynamic mode running..."

    # ── MAIN LOOP ─────────────────────────────────────────────────────────

    def run(self):
        dragging = False
        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.VIDEORESIZE:
                    self.screen_w, self.screen_h = event.w, event.h
                    self.screen = pygame.display.set_mode(
                        (self.screen_w, self.screen_h), pygame.RESIZABLE)
                    self._reposition_buttons()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    dragging = True
                    self.handle_click(event.pos, event.button)

                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging = False

                elif event.type == pygame.MOUSEMOTION and dragging:
                    # Allow drag to paint walls
                    if event.buttons[0] and self.edit_mode == 'WALL':
                        cell = self.pixel_to_cell(*event.pos)
                        if (cell and cell != self.grid.start
                                and cell != self.grid.goal):
                            self.grid.walls.add(cell)
                    elif event.buttons[2]:
                        cell = self.pixel_to_cell(*event.pos)
                        if cell:
                            self.grid.walls.discard(cell)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.start_full_search()
                    elif event.key == pygame.K_r:
                        self.reset_vis()
                    elif event.key == pygame.K_g:
                        self._ask_generate()

            # Per-frame logic
            if self.mode == 'ANIMATING':
                self.animation_tick()
            elif self.mode == 'DYNAMIC':
                self.dynamic_tick()

            # Draw
            self.screen.fill(BG_COLOR)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()


# ─────────────────────────── ENTRY ───────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.run()
