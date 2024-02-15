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
        repeated_A_star(maze, starting_coord, dest_coord, rows,cols)
        repeated_A_Star_tie(maze, starting_coord, dest_coord, rows,cols)
        repeated_Backward_A_Star(maze, starting_coord, dest_coord, rows,cols)
        # Adpative_A_star(maze3, starting_coord, dest_coord, rows,cols)
        np.savetxt('file.txt', maze, delimiter=',')

    return allMazes, allManhattan

def A_star(grid, start, end, rows, cols, backwards):
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
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), start))

    while pq:
        _, current_position = heapq.heappop(pq)
        expanded+=1
        i, j = current_position  # Update i, j to be the current position

        if current_position == end:
            if not backwards:
                return reconstruct_path(grid, prev, end), expanded  # Make sure to return the path
            else:
                return reconstruct_path_backwards(grid, prev, end), expanded
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
    return [], 0

def repeated_A_star (grid, start, end, rows, cols):
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []
    while current_start != end:
        path, expandedOnce = A_star(imaginary_mat, current_start, end, rows, cols, False)
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
            showMaze(cmap, grid)
            print(expanded)
            return p
    return []

def repeated_Backward_A_Star(grid, start, end, rows, cols):
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []
    while current_start != end:
        path, expandedOnce = A_star(imaginary_mat, end, current_start, rows, cols, True)
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
            showMaze(cmap, grid)
            print(expanded)
            return p
    return []

#Here we prefer larger g_values if the f values are the same
def A_star_tie(grid, start, end, rows, cols):
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
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), -1 * g_score[start], start)) #This is a mean heap so multiply by negative one to get largest value

    while pq:
        # print(pq)
        _, _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position
        expanded += 1
        if current_position == end:
            # print(expanded)
            return reconstruct_path(grid, prev, end), expanded  # Make sure to return the path

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
    return [], 0  # If the goal is not reached, return an empty path

def repeated_A_Star_tie(grid, start, end, rows, cols):
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    expanded = 0
    p = []
    while current_start != end:
        path, expandedOnce = A_star_tie(imaginary_mat, current_start, end, rows, cols)
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
            showMaze(cmap, grid)
            print(expanded)
            return p
    return []

def Adpative_A_star(grid, start, end, rows, cols):
    expanded = 0
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    h_score =  {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = 0
    g_score[start] = 0
    h_score[start] = manhattanDistance(start, end)
    pq = []  # Initialize the priority queue (heap)
    heapq.heappush(pq, ((0 + h_score[start]), start))

    while pq:
        file_path = 'file.txt'
# Open the file for writing ('w' mode) and write lines
        with open(file_path, 'w') as file:
            file.write(str(pq))

        _, current_position = heapq.heappop(pq)
        expanded+=1
        i, j = current_position  # Update i, j to be the current position
        h_score[i, j] = g_score[end]-g_score[i,j]
        if current_position == end:
            print(expanded)
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
                h_score[i, j] = manhattanDistance((new_i, new_j), end)
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                heapq.heappush(pq, (f_distance, (new_i, new_j)))
    return []

def reconstruct_path(grid, prev, current):
    path = []
    cmap = colors.ListedColormap(['Red','Green','Blue'])
    while current in prev:
        path.append(current)
        current = prev[current]
    path = path[::-1]
    return path

def reconstruct_path_backwards(grid, prev, current):
    path = []
    cmap = colors.ListedColormap(['Red','Green','Blue'])
    while current in prev:
        path.append(current)
        current = prev[current]
    return path

        
rows = 101
cols = 101
numMazes = 50
# mazes = genMaze(numMazes, rows, cols, allMazes)

rows = 101
cols = 101
numMazes = 1
mazes2 = genMaze(numMazes, rows, cols)
