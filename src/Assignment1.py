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
    directions = [[1,0],[0,1],[-1,0],[0,-1]]
    neighbors = []
    for dr, dc in directions:
        r, c = row+dr,col+dc
        if validRow(r) and validCol(c) and (r,c) not in visited:
            neighbors.append((r,c))
    return neighbors

def genMaze(numberOfMazes, rows, cols):
    cmap = colors.ListedColormap(['Red','Green'])
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
        allMazes.append(maze)
        allManhattan.append(genManhattan(rows, cols, dest_coord))
        cmap = colors.ListedColormap(['Red','Green'])
        plt.imshow(maze, cmap=cmap)
        plt.show()
    return allMazes, allManhattan

def genManhattan(rows, cols, dest):
    manhattan = np.zeros((rows, cols))
    for x in range(0, rows):
        for y in range(0, cols):
            manhattan[x,y] = manhattanDistance((x,y),dest)
    return manhattan  

def A_star(grid, start, end):
    #visited set of indexes visited
    visited = set()
    shortest_path = {v: float("inf") for v in grid.adjacency_list.keys()}
    direction = [[-1,0], [1,0],[0,-1],[0,1]]
    prev = {v: None for v in grid.adjacency_list.keys()}
    i,j = start[0], start[1]
    end_i, end_j = end[0], end[1]
    shortest_path[start] = 0
    queue = PriorityQueue()
    queue.add_task(0 + end_i - i + end_j - j, start)
    while queue:
        current_distance, current_position = queue.pop_task()
        visited.add(current_position)
        for d in direction:
            r,c = d[0], d[1]
            new_i = i + r
            new_j = j + c
            if(new_i < 0 or new_i >= len(grid) or new_j < 0 or new_j >= len(grid) or [new_i, new_j] in visited or grid[new_i][new_j] == 0):
                continue
            #calculate the distance from the parent to the current child plus the h(n) -> f(n) = cost + manhattan distance from the current cell
            f_distance = (current_distance + grid[new_i][new_j]) + (end_i - new_i + end_j - new_j)
            #if the distance is the shortest for this path then update the shortest distance
            if f_distance < shortest_path[[new_i, new_j]]:
                shortest_path[[new_i, new_j]] = f_distance
                prev[[new_i, new_j]] = current_position
                queue.add_task(f_distance, [new_i, new_j])
    

    #Build the path
    path = queue.LifoQueue()
    path.put(end)
    curr = end
    while curr != None:
        path.put(prev[curr])
        curr = prev[curr]
    return path







        




rows = 101
cols = 101
numMazes = 50
# mazes = genMaze(numMazes, rows, cols, allMazes)

rows = 5
cols = 5
numMazes = 1
mazes2 = genMaze(numMazes, rows, cols)
print(mazes2[0])
print(mazes2[1])