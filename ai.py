import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
import IPython.display as display
from collections import deque
import heapq
import numpy as np

# Define the maze
maze = [
    ['S','0','1','0'],
    ['1','0','1','0'],
    ['0','0','0','G']
]

# Set up global variables ROWS, COLS, and MOVES
ROWS, COLS = len(maze), len(maze[0])
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up

# Function to find the position of a symbol in the maze
def find_pos(symbol):
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == symbol:
                return r, c

# Function to check if a move is valid
def valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] != '1'

# BFS Algorithm
def bfs_maze():
    start = find_pos('S')
    goal = find_pos('G')
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current_pos, path = queue.popleft()
        if current_pos == goal:
            return path

        for dr, dc in MOVES:
            next_r, next_c = current_pos[0] + dr, current_pos[1] + dc
            next_pos = (next_r, next_c)
            if valid(next_r, next_c) and next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    return None

# DFS Algorithm
def dfs_maze():
    start = find_pos('S')
    goal = find_pos('G')
    stack = [(start, [start])]
    visited = set()

    while stack:
        current_pos, path = stack.pop()
        if current_pos == goal:
            return path

        if current_pos not in visited:
            visited.add(current_pos)
            for dr, dc in MOVES:
                next_r, next_c = current_pos[0] + dr, current_pos[1] + dc
                next_pos = (next_r, next_c)
                if valid(next_r, next_c):
                    stack.append((next_pos, path + [next_pos])) # DFS explores depth-first
    return None

# A* Algorithm
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_maze():
    start = find_pos('S')
    goal = find_pos('G')
    pq = []
    # (f_score, current_position, path)
    heapq.heappush(pq, (manhattan(start, goal), start, [start]))
    visited = set()

    while pq:
        f_score, current_pos, path = heapq.heappop(pq)
        if current_pos == goal:
            return path

        if current_pos not in visited:
            visited.add(current_pos)
            for dr, dc in MOVES:
                next_r, next_c = current_pos[0] + dr, current_pos[1] + dc
                next_pos = (next_r, next_c)
                if valid(next_r, next_c):
                    g_score = len(path) # Cost from start to current node
                    h_score = manhattan(next_pos, goal) # Heuristic cost from next node to goal
                    heapq.heappush(pq, (g_score + h_score, next_pos, path + [next_pos]))
    return None

def visualize_maze(maze, path, title):
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Convert maze characters to numerical values for colormapping
    numerical_maze = []
    for r_idx, row in enumerate(maze):
        numerical_row = []
        for c_idx, cell in enumerate(row):
            if cell == 'S':
                numerical_row.append(2)  # Start in green
            elif cell == 'G':
                numerical_row.append(3)  # Goal in red
            elif cell == '1':
                numerical_row.append(1)  # Obstacle in black
            else:
                numerical_row.append(0)  # Open path in white
        numerical_maze.append(numerical_row)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(numerical_maze, cmap=cmap, norm=norm)

    # Draw the path if provided
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, color='blue', linewidth=2)

    # Add text for 'S', 'G', '0', '1'
    for r_idx, row in enumerate(maze):
        for c_idx, cell in enumerate(row):
            ax.text(c_idx, r_idx, cell, ha='center', va='center', color='gray' if cell == '0' else 'white', fontsize=12)

    ax.set_xticks(range(len(maze[0])))
    ax.set_yticks(range(len(maze)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_title(title)

    data = io.BytesIO()
    plt.savefig(data)
    image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
    alt = title
    display.display(display.Markdown(F"![{alt}]({image})"))
    plt.close(fig)

# Calculate paths
bfs_path = bfs_maze()
dfs_path = dfs_maze()
astar_path = astar_maze()

# Display BFS visualization
print("--- BFS Path ---")
visualize_maze(maze, bfs_path, "Maze Path (BFS)")
if bfs_path:
    print(f"BFS Path Length: {len(bfs_path)}")
else:
    print("BFS: No path found.")

# Display DFS visualization
print("\n--- DFS Path ---")
visualize_maze(maze, dfs_path, "Maze Path (DFS)")
if dfs_path:
    print(f"DFS Path Length: {len(dfs_path)}")
else:
    print("DFS: No path found.")

# Display A* visualization
print("\n--- A* Path ---")
visualize_maze(maze, astar_path, "Maze Path (A*)")
if astar_path:
    print(f"A* Path Length: {len(astar_path)}")
else:
    print("A*: No path found.")