import math
import heapq
import matplotlib.pyplot as plt
import numpy as np
import json


# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.parent_k = 0  # Parent cell's depth index
        self.f = float("inf")  # Total cost of the cell (g + h)
        self.g = 0  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination


# Check if a cell is valid (within the grid)
def is_valid(i, j, k, ROW, COL, DEP):
    return i >= 0 and i < ROW and j >= 0 and j < COL and k >= 0 and k < DEP


# Check if a cell is unblocked
def is_unblocked(grid, row, col, dep):
    return grid[row][col][dep] == 1


# Check if a cell is the destination
def is_destination(i, j, k, dest):
    return i == dest[0] and j == dest[1] and k == dest[2]


# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(i, j, k, dest):
    return math.sqrt((i - dest[0]) ** 2 + (j - dest[1]) ** 2 + (k - dest[2]) ** 2)


# Trace the path from source to destination
def trace_path(cell_details, dest):
    print("The Path is ")
    path = []
    i, j, k = dest

    # Trace the path from destination to source using parent cells
    while not (
        cell_details[i][j][k].parent_i == i
        and cell_details[i][j][k].parent_j == j
        and cell_details[i][j][k].parent_k == k
    ):
        path.append((i, j, k))
        temp_i, temp_j, temp_k = (
            cell_details[i][j][k].parent_i,
            cell_details[i][j][k].parent_j,
            cell_details[i][j][k].parent_k,
        )
        i, j, k = temp_i, temp_j, temp_k

    # Add the source cell to the path
    path.append((i, j, k))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    for point in path:
        print("->", point, end=" ")
    print("\n")
    return path


def grid_creator(grid_size: tuple, obstacles: list):
    grid = np.ones(grid_size)
    if obstacles:
        grid = grid_obstacle_creator(grid, obstacles)
    return grid


def grid_obstacle_creator(grid, obstacles: list):
    """
    Create the obstacles in the grid.

    Parameters
    ----------
    obstacles: list
        The corner coordinates of each obstacle [[[i, j, k], [i, j, k]], ...].
    """

    for obstacle in obstacles:
        min_x = min(obstacle[0][0], obstacle[1][0])
        max_x = max(obstacle[0][0], obstacle[1][0])
        min_y = min(obstacle[0][1], obstacle[1][1])
        max_y = max(obstacle[0][1], obstacle[1][1])
        min_z = min(obstacle[0][2], obstacle[1][2])
        max_z = max(obstacle[0][2], obstacle[1][2])

        # Iterate through all the integer points within the ranges
        coordinates = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    coordinates.append((x, y, z))
        for coord in coordinates:
            grid[coord[0], coord[1], coord[2]] = 0

    return grid


def visualize_obstacle_bar(obstacles):
    bars_parameters = []
    for obstacle in obstacles:
        min_x = min(obstacle[0][0], obstacle[1][0])
        max_x = max(obstacle[0][0], obstacle[1][0])
        min_y = min(obstacle[0][1], obstacle[1][1])
        max_y = max(obstacle[0][1], obstacle[1][1])
        min_z = min(obstacle[0][2], obstacle[1][2])
        max_z = max(obstacle[0][2], obstacle[1][2])
        width_x = max_x - min_x
        width_y = max_y - min_y
        width_z = max_z - min_z
        bars_parameters.append((min_x, min_y, min_z, width_x, width_y, width_z))
    return bars_parameters


def collision_detection(data_list, new_i, new_j, new_k, coord_index):
    coord_index = int(coord_index)
    for data in data_list:
        if (new_i, new_j, new_k) == data[coord_index - 1]:
            return True

    return False


def a_star_algorithm(grid, src: list, dest: list, data_list: list):
    """
    Find the shortest path from the source to the destination in a 3D grid.

    Parameters
    ----------
    grid: list
        The 3D grid with obstacles (0) and free cells (1).
    src: list
        The source cell coordinates for any number of drones [(i, j, k), ...].
    dest: list
        The destination cell coordinates for any number of drones [(i, j, k), ...].
    data_list: list
        The list of coords for other drone.
    end: list
        The destination cell coordinates for any number of drones [(i, j, k), ...].
    """
    ROW, COL, DEP = len(grid), len(grid[0]), len(grid[0][0])

    if not is_valid(*src, ROW, COL, DEP) or not is_valid(*dest, ROW, COL, DEP):
        print("Source or destination is invalid")
        return []

    # Check if the source and destination are unblocked
    if not is_unblocked(grid, *src) or not is_unblocked(grid, *dest):
        print("Source or the destination is blocked")
        return []

    # Check if we are already at the destination
    if is_destination(*src, dest):
        print("We are already at the destination")
        return []

    # Initialize the closed list (visited cells)
    closed_list = [
        [[False for _ in range(DEP)] for _ in range(COL)] for _ in range(ROW)
    ]
    # Initialize the details of each cell
    cell_details = [
        [[Cell() for _ in range(DEP)] for _ in range(COL)] for _ in range(ROW)
    ]

    # Initialize the start cell details
    i, j, k = src
    cell_details[i][j][k].f = 0
    cell_details[i][j][k].g = 0
    cell_details[i][j][k].h = 0
    cell_details[i][j][k].parent_i = i
    cell_details[i][j][k].parent_j = j
    cell_details[i][j][k].parent_k = k

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j, k))

    # Initialize the flag for whether destination is found
    found_dest = False
    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)
        i, j, k = p[1], p[2], p[3]

        # Mark the cell as visited
        closed_list[i][j][k] = True

        # For each direction, check the successors
        directions = [
            (0, 1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 0, 1),
            (0, 0, -1),
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (-1, 0, 1),
            (-1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
            (0, -1, 1),
            (0, -1, -1),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]

        for di, dj, dk in directions:
            new_i, new_j, new_k = i + di, j + dj, k + dk

            # If the successor is valid, unblocked, and not visited
            if (
                is_valid(new_i, new_j, new_k, ROW, COL, DEP)
                and is_unblocked(grid, new_i, new_j, new_k)
                and not closed_list[new_i][new_j][new_k]
                and not collision_detection(
                    data_list, new_i, new_j, new_k, cell_details[i][j][k].g
                )
            ):
                # If the successor is the destination
                if is_destination(new_i, new_j, new_k, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j][new_k].parent_i = i
                    cell_details[new_i][new_j][new_k].parent_j = j
                    cell_details[new_i][new_j][new_k].parent_k = k
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)
                    found_dest = True
                    return path
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j][k].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, new_k, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if (
                        cell_details[new_i][new_j][new_k].f == float("inf")
                        or cell_details[new_i][new_j][new_k].f > f_new
                    ):
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j, new_k))
                        # Update the cell details
                        cell_details[new_i][new_j][new_k].f = f_new
                        cell_details[new_i][new_j][new_k].g = g_new
                        cell_details[new_i][new_j][new_k].h = h_new
                        cell_details[new_i][new_j][new_k].parent_i = i
                        cell_details[new_i][new_j][new_k].parent_j = j
                        cell_details[new_i][new_j][new_k].parent_k = k

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")


def visualize(start, end, path, obstacles: list = []):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    number_of_drones = 0
    colors = ["blue", "green", "purple", "yellow", "orange", "pink", "brown"]

    if obstacles:
        bars_parameters = visualize_obstacle_bar(obstacles)
        for bar in bars_parameters:
            ax.bar3d(*bar, color="red", alpha=0.5, zorder=1)

    for src, dest in zip(start, end):
        if path[number_of_drones]:
            ax.scatter(src[0], src[1], src[2], color=colors[number_of_drones])
            ax.scatter(dest[0], dest[1], dest[2], color=colors[number_of_drones])
            z = []
            y = []
            x = []
            for i in path[number_of_drones]:
                x.append(i[0])
                y.append(i[1])
                z.append(i[2])

            ax.plot(
                x, y, z, color=colors[number_of_drones], zorder=10 + number_of_drones
            )
            number_of_drones += 1

    plt.show()


def a_star_search(start, end, grid_size, obstacles=[]):
    """
    Run the A* search algorithm for a 3D space.

    Parameters
    ----------
    start: tuple
        The initial coordinates [(i, j, k), ...]
    end: tuple
        The final coordinates [(i, j, k), ...]
    grid_size: tuple
        The size of the grid (row, column, depth).
    obstacles: list
        The corner coordinates of each obstacle [[[i, j, k], [i, j, k]], ...].
    """
    data_list = []
    grid = grid_creator(grid_size, obstacles)
    for src, dest in zip(start, end):
        data = a_star_algorithm(grid, src, dest, data_list)
        data_list.append(data)
    visualize(start, end, data_list, obstacles)


if __name__ == "__main__":
 # Define the grid size
    grid_size = (60, 60, 60)

    # Define the source and destination coordinates for multiple drones
    start = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ]
    end = [
        [58, 58, 58],
        [57, 57, 57],
        [56, 56, 56]
    ]

    # Define more complex obstacles within the grid
    obstacles = [
        [[5, 5, 5], [10, 10, 10]],
        [[5,10,10], [10,15,15]],
    ]


    a_star_search(start, end, grid_size, obstacles)
