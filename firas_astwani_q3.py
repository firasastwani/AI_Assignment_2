
import math
import heapq
from typing import List, Tuple, Optional, Set


def find_shortest_path(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[bool]]) -> Optional[List[Tuple[int, int]]]:
    """
    Find the shortest path from start to goal on a 2D grid with obstacles using A* search.
    
    Args:
        start: Starting point as (row, column) tuple
        goal: Goal point as (row, column) tuple
        grid: 2D list where False = empty space, True = obstacle
    
    Returns:
        List of points along the shortest path, or None if no path exists
    """
    # Validate inputs
    if not _is_valid_point(start, grid) or not _is_valid_point(goal, grid):
        return None
    
    # Check if start or goal is on an obstacle
    if grid[start[0]][start[1]] or grid[goal[0]][goal[1]]:
        return None
    
    # If start and goal are the same
    if start == goal:
        return [start]
    
    # Initialize A* search
    open_set = []  # Priority queue: (f_cost, g_cost, point, path)
    heapq.heappush(open_set, (0, 0, start, [start]))
    
    closed_set: Set[Tuple[int, int]] = set()
    
    # 8-directional movement: up, down, left, right, and 4 diagonals
    directions = [
        (-1, 0),   # up
        (1, 0),    # down
        (0, -1),   # left
        (0, 1),    # right
        (-1, -1),  # up-left
        (-1, 1),   # up-right
        (1, -1),   # down-left
        (1, 1)     # down-right
    ]
    
    while open_set:
        f_cost, g_cost, current, path = heapq.heappop(open_set)
        
        # Skip if we've already processed this point
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Check if we've reached the goal
        if current == goal:
            return path
        
        # Explore all 8 neighboring directions
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Skip if neighbor is invalid or on obstacle
            if not _is_valid_point(neighbor, grid) or grid[neighbor[0]][neighbor[1]]:
                continue
            
            # Skip if already processed
            if neighbor in closed_set:
                continue
            
            # Calculate movement cost (1 for cardinal directions, sqrt(2) for diagonals)
            if dr != 0 and dc != 0:  # Diagonal movement
                move_cost = math.sqrt(2)
            else:  # Cardinal movement
                move_cost = 1
            
            new_g_cost = g_cost + move_cost
            
            # Calculate heuristic (Euclidean distance to goal)
            heuristic = _euclidean_distance(neighbor, goal)
            new_f_cost = new_g_cost + heuristic
            
            # Create new path
            new_path = path + [neighbor]
            
            # Add to open set
            heapq.heappush(open_set, (new_f_cost, new_g_cost, neighbor, new_path))
    
    # No path found
    return None


def _is_valid_point(point: Tuple[int, int], grid: List[List[bool]]) -> bool:
    """
    Check if a point is within the grid boundaries.
    
    Args:
        point: (row, column) tuple
        grid: 2D grid
    
    Returns:
        True if point is valid, False otherwise
    """
    row, col = point
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def _euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as (row, column)
        point2: Second point as (row, column)
    
    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def test_examples():
    """Test the function with the provided examples."""
    print("Testing Grid Pathfinding Examples")
    print("=" * 40)
    
    # Example 1
    print("Example 1:")
    grid1 = [[False, False, False],
             [False, True, False],
             [False, False, False]]
    result1 = find_shortest_path((0, 0), (2, 1), grid1)
    print(f"Grid: {grid1}")
    print(f"Start: (0, 0), Goal: (2, 1)")
    print(f"Result: {result1}")
    print(f"Expected: [(0, 0), (1, 0), (2, 1)]")
    print()
    
    # Example 2
    print("Example 2:")
    grid2 = [[False, True, False],
             [False, True, False],
             [False, True, False]]
    result2 = find_shortest_path((0, 0), (0, 2), grid2)
    print(f"Grid: {grid2}")
    print(f"Start: (0, 0), Goal: (0, 2)")
    print(f"Result: {result2}")
    print(f"Expected: None")
    print()
    
    # Additional test cases
    print("Additional Test Cases:")
    
    # Test 3: Simple diagonal path
    grid3 = [[False, False],
             [False, False]]
    result3 = find_shortest_path((0, 0), (1, 1), grid3)
    print(f"Diagonal path test: {result3}")
    
    # Test 4: Start and goal are the same
    result4 = find_shortest_path((0, 0), (0, 0), grid3)
    print(f"Same start/goal test: {result4}")
    
    # Test 5: Start on obstacle
    result5 = find_shortest_path((0, 1), (1, 1), grid1)
    print(f"Start on obstacle test: {result5}")


def visualize_path(grid: List[List[bool]], path: Optional[List[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]):
    """
    Visualize the grid and path for debugging.
    
    Args:
        grid: 2D grid with obstacles
        path: List of points in the path
        start: Starting point
        goal: Goal point
    """
    if not path:
        print("No path found")
        return
    
    # Create visualization grid
    rows, cols = len(grid), len(grid[0])
    vis_grid = [['.' for _ in range(cols)] for _ in range(rows)]
    
    # Mark obstacles
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]:
                vis_grid[i][j] = '#'
    
    # Mark path
    for i, (r, c) in enumerate(path):
        if i == 0:
            vis_grid[r][c] = 'S'  # Start
        elif i == len(path) - 1:
            vis_grid[r][c] = 'G'  # Goal
        else:
            vis_grid[r][c] = '*'  # Path
    
    # Print visualization
    print("Grid Visualization:")
    for row in vis_grid:
        print(' '.join(row))
    print()


if __name__ == "__main__":
    test_examples()
    
    # Additional visualization test
    print("\nPath Visualization Example:")
    grid_viz = [[False, False, False],
                [False, True, False],
                [False, False, False]]
    path_viz = find_shortest_path((0, 0), (2, 1), grid_viz)
    visualize_path(grid_viz, path_viz, (0, 0), (2, 1))
