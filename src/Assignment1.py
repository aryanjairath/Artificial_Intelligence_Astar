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
        # showMaze(cmap,maze)
        allMazes.append(maze)
        print(adaptive_A_star(maze, starting_coord, dest_coord, rows,cols))
        print(repeated_A_star(maze, starting_coord, dest_coord, rows,cols))
        # print(repeated_Backward_A_Star(maze, starting_coord, dest_coord, rows,cols))
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
            # showMaze(cmap, grid)
            return p, expanded
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
    heapq.heappush(pq, ((0 + manhattanDistance(start, end)), -1 * g_score[start], start))

    while pq:
        # print(pq)
        _, _, current_position = heapq.heappop(pq)
        i, j = current_position  # Update i, j to be the current position
        expanded += 1
        if current_position == end:
            # print(expanded)
            return reconstruct_path(prev, end), expanded  # Make sure to return the path

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
            return p, expanded
    return []
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


def A_star_Adaptive(grid, start, end, rows, cols, backwards, g_score, search, pathcost, counter, h_score):
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
                return reconstruct_path(prev, end), g_score[current_position], expanded# Make sure to return the path
            else:
                return reconstruct_path_backwards(prev, end), g_score[current_position], expanded
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

        path, total_traveled, expandedOnce = A_star_Adaptive(imaginary_mat, current_start, end, rows, cols, False, g_score, search, pathcost, counter, h_score)
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
# def Adpative_A_star(grid, start, end, rows, cols):
#     current_start = start
#     h = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             h[i, j] = abs(end[0] - i) + abs(end[1] - j)
#     imaginary_mat = np.ones((rows,cols))
#     expanded = 0
#     p = []
#     while current_start != end:
#         path, expandedOnce, h = A_star2(imaginary_mat, current_start, end, rows, cols, h)
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
#             cmap = colors.ListedColormap(['Red','Green', 'Blue'])
#             for coord in p:
#                 grid[coord[0],[coord[1]]] = 2
#             # showMaze(cmap, grid)
#             return p, expanded
#     return []    
# 
# def A_star2(grid, start, end, rows, cols, h):
#     expanded = 0
#     visited = set()
#     g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
#     prev = {(i, j): None for i in range(rows) for j in range(cols)}
#     pq = []  # Priority queue
#     g_score[start] = 0
#     heapq.heappush(pq, (h[start[0], start[1]], start))  # Use h for the initial heuristic
#     while pq:
#         _, current_position = heapq.heappop(pq)
#         expanded += 1
#         if current_position == end:
#             for node in visited:
#                 h[node[0], node[1]] = g_score[end] - g_score[node]  # Update h using the provided formula
#             return reconstruct_path(prev, end), expanded, h
#         visited.add(current_position)
#         i, j = current_position

#         for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
#             new_i, new_j = i + d[0], j + d[1]
#             if not (validRow(new_i) and validCol(new_j)) or (new_i, new_j) in visited or grid[new_i][new_j] == 0:
#                 continue
#             tentative_g = g_score[current_position] + 1
#             if tentative_g < g_score.get((new_i, new_j), float('inf')):
#                 prev[(new_i, new_j)] = current_position
#                 g_score[(new_i, new_j)] = tentative_g
#                 f_score = tentative_g + h[new_i, new_j]  # Use updated h for computing f
#                 heapq.heappush(pq, (f_score, (new_i, new_j)))

#     return [], 0, 0


def reconstruct_path(prev, current):
    path = []
    cmap = colors.ListedColormap(['Red','Green','Blue'])
    while current in prev:
        path.append(current)
        current = prev[current]
    path = path[::-1]
    return path

def reconstruct_path_backwards(prev, current):
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

rows = 150
cols = 150
numMazes = 1
mazes2 = genMaze(numMazes, rows, cols)
