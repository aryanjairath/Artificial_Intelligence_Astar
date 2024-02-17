import random
import queue
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

        adaptive_A_star(maze, starting_coord, dest_coord, rows,cols)
        # np.savetxt('file.txt', maze, delimiter=',')

    return allMazes, allManhattan

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


#updates h at state s
def update_h(state, search, pathcost, end, g_score, counter, h_score):
    #this updates the h_score based on the previous iteration of a_star
    if search[state] != counter and search[state] != 0:
        if g_score[state] + h_score[state] < pathcost[search[state]]:
            h_score[state] = pathcost[search[state]] - g_score[state]
        h_score[state] = max(h_score[state], manhattanDistance(state, end))
        g_score[state] = float('inf')
    
    # this is for the first time A_star is called
    elif search[state] == 0:
        g_score[state] = float('inf')
        h_score[state] = manhattanDistance(state, end)
    search[state] = counter


def A_star_2(grid, start, end, rows, cols, backwards, g_score, search, pathcost, counter, h_score):
    visited = set()
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    i,j = start[0], start[1]
    f_score[start] = 0
    g_score[start] = 0
    pq = []  # Initialize the priority queue (heap)
    expanded = 0
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), start))
    #
    while pq:
        _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position
        expanded+=1
        if current_position == end:
            if not backwards:
                return reconstruct_path(grid, prev, end), g_score[current_position], expanded# Make sure to return the path
            else:
                return reconstruct_path_backwards(grid, prev, end), g_score[current_position], expanded
        visited.add(current_position)
        for d in direction:
            r, c = d[0], d[1]
            new_i, new_j = i + r, j + c  # Correctly calculate new_i and new_j

            if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
                continue  # Check grid boundaries and visited or blocked cells
            update_h((new_i, new_j), search, pathcost, end, g_score, counter, h_score)

            # Calculate the f_score for the neighbor
            f_distance = g_score[(i, j)] + 1 + manhattanDistance((new_i, new_j), end)
            if g_score[current_position] + 1 < g_score[(new_i, new_j)]:
                prev[(new_i, new_j)] = current_position  # Update the prev pointer
                f_score[(new_i, new_j)] = f_distance
                g_score[(new_i, new_j)] = g_score[current_position] + 1
                heapq.heappush(pq, (f_distance, (new_i, new_j)))
    return [], 0, 0

def adaptive_A_star (grid, start, end, rows, cols):
    current_start = start
    imaginary_mat = np.ones((rows,cols))
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    h_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    pathcost = [0 for i in range(rows) for j in range(cols)]
    counter = 1
    search = {(i, j): 0 for i in range(rows) for j in range(cols)}
    expanded = 0
    p = []
    while current_start != end:
        update_h(current_start, search, pathcost, end, g_score, counter, h_score)
        update_h(end, search, pathcost, end, g_score, counter, h_score)

        path, total_traveled, expandedOnce = A_star_2(imaginary_mat, current_start, end, rows, cols, False, g_score, search, pathcost, counter, h_score)
        expanded += expandedOnce
        if not path:
            break
        for step in path:
            if grid[step] == 0:
                imaginary_mat[step] = 0
                counter += 1
                break
            else:
                current_start = step
                p.append(step)
                pathcost[counter] = total_traveled
        if current_start == end:
            cmap = colors.ListedColormap(['Red','Green', 'Blue'])
            for coord in p:
                grid[coord[0],[coord[1]]] = 2
            showMaze(cmap, grid)
            return p, expanded
    return [], 0

rows = 101
cols = 101
numMazes = 1
mazes2 = genMaze(numMazes, rows, cols)