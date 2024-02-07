import random
import numpy as np
import matplotlib 
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

def genMaze(numberOfMazes, rows, cols, allMazes):
    cmap = colors.ListedColormap(['Red','Green'])
    for _ in numberOfMazes:
        maze = np.zeros((rows,cols))
        starting_row = random.randint(0,rows-1) 
        starting_col = random.randint(0,cols-1)
        visited = set()
        stack = []
        stack.append((starting_row, starting_col))
        visited.add((starting_row, starting_col))
        maze[starting_row,starting_col] = 1

        while len(visited) < rows*cols:
            if stack:
                curr_row, curr_col = stack[-1]
            else:
                unvisited_cells = [(r,c) for r in range(rows) for c in range(cols) if (r,c) not in visited]
                curr_row, curr_col = random.choice(unvisited_cells)
                stack.append((curr_row,curr_col))
                visited.add((curr_row,curr_col))
                maze[curr_row,curr_col] = 1
                
            neighbors = get_unvisited_neighbors(maze, curr_row, curr_col, visited)
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
        allMazes.append(maze)
        print(maze)
        cmap = colors.ListedColormap(['Red','Green'])
        plt.imshow(maze, cmap=cmap)
        plt.show()
    return allMazes

rows = 101
cols = 101
numMazes = 50
# mazes = genMaze(numMazes, rows, cols, allMazes)


rows = 5
cols = 5
numMazes = 7
mazes2 = genMaze(numMazes, rows, cols, [])
