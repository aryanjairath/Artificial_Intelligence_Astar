import random
import queue
import PriorityQueue
import numpy as np
import matplotlib 
import heapq
import math
from matplotlib import colors
from matplotlib import pyplot as plt


def validRow(row):
    return 0 <= row < rows

def validCol(col):
    return 0 <= col < cols

def manhattanDistance(curr, goal):
    return abs(goal[1]-curr[1])+abs(goal[0]-curr[0])

def get_unvisited_neighbors(row, col, visited):
    direction = [[1,0],[0,1],[-1,0],[0,-1]]
    neighbors = []
    for dr, dc in direction:
        r, c = row+dr,col+dc
        if validRow(r) and validCol(c) and (r,c) not in visited:
            neighbors.append((r,c))
    return neighbors

def showMaze(cmap, maze):
    plt.figure(figsize=(6.7,6.7))
    plt.imshow(maze, cmap=cmap)
    plt.show()

def genMaze(numberOfMazes, rows, cols):
    allMazes = []
    allManhattan = []
    for _ in range(numberOfMazes):
        maze = np.zeros((rows,cols))
        starting_coord = (random.randint(0,rows-1), random.randint(0,cols-1))
        dest_coord = (random.randint(0,rows-1), random.randint(0,cols-1))
        print(starting_coord)
        print(dest_coord)
        visited = set()
        stack = []
        stack.append(starting_coord)
        visited.add(starting_coord)
        maze[starting_coord[0],starting_coord[1]] = 1

        while len(visited) < rows*cols:
            if stack:
                curr_row, curr_col = stack[-1]
            else: #Dead end check
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
            else: #No neighbors!
                stack.pop()
        maze[dest_coord] = 1
        cmap = colors.ListedColormap(['Red','Green'])
        showMaze(cmap,maze)
        allMazes.append(maze)
        print(Backward_A_star(maze, starting_coord, dest_coord, rows,cols))
    return allMazes, allManhattan

def A_star(grid, start, end, rows, cols):
    
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = 0
    g_score[start] = 0
    pq = []  # Initialize the priority queue (heap)
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), start))

    while pq:
        _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position

        if current_position == end:
            return reconstruct_path(grid, prev, end)  # Make sure to return the path

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
                heapq.heappush(pq, (f_distance, (new_i, new_j)))
    return []  # If the goal is not reached, return an empty path
    
def Backward_A_star(grid, start, end, rows, cols):
   
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = end[0], end[1]
    f_score[end] = 0
    g_score[end] = 0
    pq = []  # Initialize the priority queue (heap)
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), end))

    while pq:
        _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position

        if current_position == start:
            return reconstruct_path(grid, prev, start)  # Make sure to return the path

        visited.add(current_position)
        for d in direction:
            r, c = d[0], d[1]
            new_i, new_j = i + r, j + c  # Correctly calculate new_i and new_j

            if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
                continue  # Check grid boundaries and visited or blocked cells

            # Calculate the f_score for the neighbor
            f_distance = g_score[(i, j)] + 1 + manhattanDistance((new_i, new_j), start)
            if g_score[current_position] + 1 < g_score[(new_i, new_j)]:
                prev[(new_i, new_j)] = current_position  # Update the prev pointer
                f_score[(new_i, new_j)] = f_distance
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                heapq.heappush(pq, (f_distance, (new_i, new_j)))
    return []  # If the goal is not reached, return an empty path

#Here we prefer larger g_values if the f values are the same
def A_star_tie(grid, start, end, rows, cols):
    #visited set of indexes visited
    
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = 0
    g_score[start] = 0
    pq = []  # Initialize the priority queue (heap)
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), -1 * g_score[start], start))

    while pq:
        _, _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position

        if current_position == end:
            return reconstruct_path(grid, prev, end)  # Make sure to return the path

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
                heapq.heappush(pq, (f_distance, -1 * g_score[(new_i, new_j)], (new_i, new_j)))
    return []  # If the goal is not reached, return an empty path

def reconstruct_path(grid, prev, current):
    path = []
    cmap = colors.ListedColormap(['Red','Green','Blue'])
    while current in prev:
        if current:
            grid[current[0], current[1]] = 2
        path.append(current)
        current = prev[current]
    showMaze(cmap, grid)
    path = path[::-1]
    return path

        
rows = 101
cols = 101
numMazes = 50
# mazes = genMaze(numMazes, rows, cols, allMazes)

rows = 5
cols = 5
numMazes = 1
mazes2 = genMaze(numMazes, rows, cols)

