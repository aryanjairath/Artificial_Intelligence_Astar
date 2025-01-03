import random
import os
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

# Indicates whether we want to use our own heap class or the python provided one
use_custom_heap = True
# Seed for testing custom binary heap implementation, set to None to use random seed
seed = None

if use_custom_heap:
    print("Using custom binary heap")
    import binary_heap
    heapq = binary_heap.binary_heap()  
else:
    print("Using Python's heapq")
    import heapq

# Set to True if figures are wanted
displayMazes = True

# Indicates if we want to write mazes to a file
writeMazes = False
writePaths = True
assert not (writeMazes and writePaths), "Cannot write both mazes and paths"

#Set to True if break ties by larger g values
larger_g = True

# Below variables keeps track of how many expanded cells are visited
repeat = 0
backward = 0
adapt = 0
ties = 0

# Keeps track of the maze number
mazeNumber = 0

# North, East, South, West
cardinal_directions = [[1,0],[0,1],[-1,0],[0,-1]]

# Determines if a row is valid
def validRow(row):
    return 0 <= row < rows

# Determines if a column is valid
def validCol(col):
    return 0 <= col < cols

# Returns Manhattan Distance -- shortest distance between two points (returns non integer value)
def manhattanDistance(curr, goal):
    return abs(goal[1] - curr[1]) + abs(goal[0] - curr[0])

# Returns set of neighbors that have not been visited at point (row, col)
def get_unvisited_neighbors(row, col, visited):
    neighbors = []
    for dr, dc in cardinal_directions:
        r, c = row + dr, col + dc
        if validRow(r) and validCol(c) and (r,c) not in visited:
            neighbors.append((r,c))
    return neighbors

# Returns list of nodes representing shortest path from the current node to the starting node
def reconstruct_path(prev, current):
    path = []
    while current in prev:
        path.append(current)
        current = prev[current]
    path = path[::-1]
    return path

# Same as reconstruct_path but omits reversal of path
def reconstruct_path_backwards(prev, current):
    path = []
    while current in prev:
        path.append(current)
        current = prev[current]
    return path

# Prints maze
def showMaze(cmap, maze, title='Generated Maze'):
    plt.figure(figsize=(6.7,6.7))
    plt.imshow(maze, cmap=cmap)
    plt.title(title)
    plt.show()

# Generates mazes
def genMaze(numberOfMazes, rows, cols):
    global backward
    global adapt
    global ties
    global repeat
    global mazeNumber
    repeat = 0
    backward = 0
    adapt = 0
    ties = 0
    allMazes = []
    for _ in range(numberOfMazes):
        maze = np.zeros((rows,cols))
        
        # Use seed for testing custom binary heap implementation
        if seed:
            random.seed(seed)

        starting_coord = (random.randint(0,rows-1), random.randint(0,cols-1))
        dest_coord = (random.randint(0,rows-1), random.randint(0,cols-1))
        print("Start:", starting_coord)
        print("Goal:", dest_coord)
        visited = set()
        stack = []
        stack.append(starting_coord)
        visited.add(starting_coord)
        maze[starting_coord[0],starting_coord[1]] = 1

        while len(visited) < rows*cols:
            if stack:
                curr_row, curr_col = stack[-1]
            else:
                unvisited_cells = [(r,c) for r in range(rows) for c in range(cols) if (r,c) not in visited]
                curr_row, curr_col = random.choice(unvisited_cells)
                stack.append((curr_row,curr_col))
                visited.add((curr_row,curr_col))
                maze[curr_row,curr_col] = 1
                
            neighbors = get_unvisited_neighbors(curr_row, curr_col, visited)
            if neighbors:
                next_row, next_col = random.choice(neighbors)
                visited.add((next_row, next_col))
                if random.random() < 0.7:
                    maze[next_row,next_col] = 1
                    stack.append((next_row,next_col))
                else:
                    maze[next_row, next_col] = 0
            else:
                stack.pop()
        maze[dest_coord] = 1
        cmap = colors.ListedColormap(['Red','Green'])
        displayMazes and showMaze(cmap, maze, "Initial Maze")

        if writePaths or writeMazes:
            if not os.path.exists('mazes'):
                os.makedirs('mazes')

            with open('mazes/' + str(mazeNumber + 1) + '_initial.txt', 'w') as f:
                for row in maze:
                    f.write(' '.join([str(int(cell)) for cell in row]) + '\n')

        allMazes.append(maze)

        # Run all algorithms
        repeated_A_Star_tie(maze, starting_coord, dest_coord, rows,cols)
        repeated_Backward_A_Star(maze, starting_coord, dest_coord, rows,cols)
        Adaptive_A_star(maze, starting_coord, dest_coord, rows,cols)
        print("# of Expanded Nodes for --> Repated Forward A*:", ties, "Backward A*:", backward, "Adaptive A*:", adapt)
        
        mazeNumber += 1

    return allMazes

def compute_path(grid, start, end, rows, cols, backwards, larger_g):
    expanded = 0
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = manhattanDistance(start, end)
    g_score[start] = 0
    pq = []  # Initialize the priority queue (heap)
    cc = rows*cols
    heapq.heappush(pq, ((cc*f_score[start]), start))

    while pq:
        _, current_position = heapq.heappop(pq)
        expanded+=1
        i, j = current_position  # Update i, j to be the current position

        if current_position == end:
            if not backwards:
                return reconstruct_path(prev, end), expanded  # Make sure to return the path
            else:
                return reconstruct_path_backwards(prev, end), expanded
        visited.add(current_position)
        for d in direction:
            r, c = d[0], d[1]
            new_i, new_j = i + r, j + c  # Correctly calculate new_i and new_j

            if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
                continue  # Check grid boundaries and visited or blocked cells

            # Calculate the f_score for the neighbor
            f_distance = g_score[(i, j)] + 1 + manhattanDistance((new_i, new_j), end)
            if g_score[current_position] + 1 < g_score[(new_i, new_j)]:
                prev[(new_i, new_j)] = current_position  # Update the prev pointer
                f_score[(new_i, new_j)] = f_distance
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                if larger_g:
                    heapq.heappush(pq, (cc * f_score[(new_i, new_j)] - g_score[(new_i, new_j)], (new_i, new_j)))
                else:
                    heapq.heappush(pq, (cc * f_score[(new_i, new_j)] + g_score[(new_i, new_j)], (new_i, new_j)))

    return [], 0

# Take the grid, source, dest, and dimensions and perform adaptive A* search
def compute_path_adaptive(grid, start, end, rows, cols, h):
    global adapt
    expanded = 0
    visited = set()
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    pq = []  # Priority queue
    g_score[start] = 0
    cc = rows * cols
    heapq.heappush(pq, (cc*manhattanDistance(start,end) - g_score[start], start))  # Use h for the initial heuristic
    while pq:
        _, current_position = heapq.heappop(pq)
        expanded += 1
        if current_position == end:
            adapt+=expanded
            for node in visited:
                h[node[0], node[1]] = g_score[end] - g_score[node]  # Update h using the provided formula
            return reconstruct_path(prev, end), expanded, h
        visited.add(current_position)
        i, j = current_position

        for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            new_i, new_j = i + d[0], j + d[1]
            if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
                continue

             
            if g_score[current_position] + 1 < g_score.get((new_i, new_j), float('inf')):
                prev[(new_i, new_j)] = current_position
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                f_score = g_score[current_position] + 1 + h[new_i, new_j]  # Use updated h for computing f
                heapq.heappush(pq, (cc*f_score - g_score[(new_i, new_j)], (new_i, new_j)))

    return [], 0, []

#Take the grid, source, dest, and dimensions and perform normal A* search
# def repeated_A_star (grid, start, end, rows, cols):
#     global repeat
#     current_start = start
#     imaginary_mat = np.ones((rows,cols))
#     expanded = 0
#     p = []
#     while current_start != end:
#         path, expandedOnce = compute_path(imaginary_mat, current_start, end, rows, cols, False)
#         expanded += expandedOnce
#         if not path:
#             break
#         for step in path:
#             if grid[step] == 0:
#                 imaginary_mat[step] = 0
#                 break
#             else:
#                 current_start = step
#                 p.append(step)
#         if current_start == end:
#             repeat += expanded
#             cmap = colors.ListedColormap(['Red','Green', 'Blue'])
#             for coord in p:
#                 grid[coord[0],[coord[1]]] = 2
#             displayMazes and showMaze(cmap, grid, "Forwards A*")
#             return p, expanded
        
#     return [], 0

# Take the grid, source, dest, and dimensions and perform backwards A* search
def repeated_Backward_A_Star(grid, start, end, rows, cols):
    global backward
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []
    while current_start != end:
        path, expandedOnce = compute_path(imaginary_mat, end, current_start, rows, cols, True, larger_g)
        expanded += expandedOnce
        if not path:
            break
        for step in path:
            if grid[step] == 0:
                imaginary_mat[step] = 0
                break
            else:
                current_start = step
                p.append(step)
        if current_start == end:
            backward += expanded
            cmap = colors.ListedColormap(['Red','Green', 'Blue'])
            for coord in p:
                grid[coord[0],[coord[1]]] = 2

            displayMazes and showMaze(cmap, grid, "Backwards A*")
            if writeMazes:
                with open('mazes/' + str(mazeNumber + 1) + '_backwards.txt', 'w') as f:
                    for row in grid:
                        f.write(' '.join([str(int(cell)) for cell in row]) + '\n')
            if writePaths:
                with open('mazes/' + str(mazeNumber + 1) + '_backwards.txt', 'w') as f:
                    for coord in p:
                        f.write('(' + str(coord[0]) + ', ' + str(coord[1]) + ')\n')
                    

            return p, expanded
        
    return [], 0

# Here we prefer larger g_values if the f values are the same
def compute_path_ties(grid, start, end, rows, cols):
    #visited set of indexes visited
    expanded = 0
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = 0
    g_score[start] = 0
    pq = []  # Initialize the priority queue (heap)
    cc = rows * cols
    heapq.heappush(pq, ((cc * manhattanDistance(start, end) + g_score[start]), start))
    while pq:
        _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position
        expanded += 1
        if current_position == end:
            return reconstruct_path(prev, end), expanded  # Make sure to return the path

        visited.add(current_position)
        for d in direction:
            r, c = d[0], d[1]
            new_i, new_j = i + r, j + c  # Correctly calculate new_i and new_j

            if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
                continue  # Check grid boundaries and visited or blocked cells

            # Calculate the f_score for the neighbor
            if g_score[current_position] + 1 < g_score[(new_i, new_j)]:
                prev[(new_i, new_j)] = current_position  # Update the prev pointer
                f_score[(new_i, new_j)] = g_score[current_position] + 1 + manhattanDistance((new_i, new_j), end)
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                heapq.heappush(pq, (cc * f_score[(new_i, new_j)] - g_score[(new_i, new_j)], (new_i, new_j)))

    return [], 0  # If the goal is not reached, return an empty path

def repeated_A_Star_tie(grid, start, end, rows, cols):
    global ties
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []

    while current_start != end:
        path, expandedOnce = compute_path(imaginary_mat, current_start, end, rows, cols, False, larger_g)
        expanded += expandedOnce
        if not path:
            break
        for step in path:
            if grid[step] == 0:
                imaginary_mat[step] = 0
                break
            else:
                current_start = step
                p.append(step)
        if current_start == end:
            ties += expanded
            cmap = colors.ListedColormap(['Red','Green', 'Blue'])
            for coord in p:
                grid[coord[0],[coord[1]]] = 2
            
            displayMazes and showMaze(cmap, grid, "Tie Breaking A*")
            if writeMazes:
                with open('mazes/' + str(mazeNumber + 1) + '_ties.txt', 'w') as f:
                    for row in grid:
                        f.write(' '.join([str(int(cell)) for cell in row]) + '\n')

            if writePaths:
                with open('mazes/' + str(mazeNumber + 1) + '_ties.txt', 'w') as f:
                    for coord in p:
                        f.write('(' + str(coord[0]) + ', ' + str(coord[1]) + ')\n')

            return p, expanded
        
    return [], 0

def Adaptive_A_star(grid, start, end, rows, cols):
    current_start = start
    h = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            h[i, j] = manhattanDistance((i,j), end)
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []

    while current_start != end:
        path, expandedOnce, h = compute_path_adaptive(imaginary_mat, current_start, end, rows, cols, h)
        expanded += expandedOnce
        if not path:
            break
        for step in path:
            if grid[step] == 0:
                imaginary_mat[step] = 0
                break
            else:
                current_start = step
                p.append(step)
        if current_start == end:
            cmap = colors.ListedColormap(['Red','Green', 'Blue'])
            for coord in p:
                grid[coord[0],[coord[1]]] = 2

            displayMazes and showMaze(cmap, grid, "Adaptive A*")
            if writeMazes:
                with open('mazes/' + str(mazeNumber + 1) + '_adaptive.txt', 'w') as f:
                    for row in grid:
                        f.write(' '.join([str(int(cell)) for cell in row]) + '\n')

            if writePaths:
                with open('mazes/' + str(mazeNumber + 1) + '_adaptive.txt', 'w') as f:
                    for coord in p:
                        f.write('(' + str(coord[0]) + ', ' + str(coord[1]) + ')\n')

            return p, expanded
        
    return [], 0    

rows = 101
cols = 101
numMazes = 50
# numMazes = 1
mazes = genMaze(numMazes, rows, cols)
