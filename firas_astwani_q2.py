# Pulled algo overviews from teams and used AI to adjust them

import csv
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set
import heapq


class Building:
    """Represents a building on campus with coordinates and name."""
    
    def __init__(self, building_id: int, x_coord: int, y_coord: int, name: str):
        self.id = building_id
        self.x = x_coord
        self.y = y_coord
        self.name = name
    
    def __str__(self):
        return f"Building {self.id}: {self.name} ({self.x}, {self.y})"


class CampusGraph:
    """Graph representation of the campus with buildings and walkways."""
    
    def __init__(self):
        self.buildings: Dict[int, Building] = {}
        self.adjacency_list: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    
    def add_building(self, building: Building):
        """Add a building to the campus graph."""
        self.buildings[building.id] = building
    
    def add_walkway(self, building1_id: int, building2_id: int, travel_time: int):
        """Add a bidirectional walkway between two buildings."""
        self.adjacency_list[building1_id].append((building2_id, travel_time))
        self.adjacency_list[building2_id].append((building1_id, travel_time))
    
    def get_neighbors(self, building_id: int) -> List[Tuple[int, int]]:
        """Get neighboring buildings with their travel times."""
        return self.adjacency_list[building_id]
    
    def heuristic_distance(self, building1_id: int, building2_id: int) -> int:
        """
        Calculate Euclidean distance between two buildings as heuristic.
        Returns rounded integer distance.
        """
        building1 = self.buildings[building1_id]
        building2 = self.buildings[building2_id]
        
        distance = math.sqrt(
            (building1.x - building2.x) ** 2 + 
            (building1.y - building2.y) ** 2
        )
        return round(distance)
    
    def load_from_files(self, buildings_file: str, walkways_file: str):
        """Load campus data from CSV files."""
        # Load buildings
        with open(buildings_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                building = Building(
                    int(row['building_id']),
                    int(row['x_coord']),
                    int(row['y_coord']),
                    row['building_name']
                )
                self.add_building(building)
        
        # Load walkways
        with open(walkways_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_walkway(
                    int(row['building1_id']),
                    int(row['building2_id']),
                    int(row['travel_time'])
                )


class SearchResult:
    """Container for search algorithm results."""
    
    def __init__(self, path: List[int], total_time: int, nodes_expanded: int, algorithm_name: str):
        self.path = path
        self.total_time = total_time
        self.nodes_expanded = nodes_expanded
        self.algorithm_name = algorithm_name
    
    def __str__(self):
        return f"{self.algorithm_name}: Time={self.total_time}, Nodes={self.nodes_expanded}, Path={self.path}"


class CampusNavigator:
    """Main class implementing various search algorithms for campus navigation."""
    
    def __init__(self, campus_graph: CampusGraph):
        self.graph = campus_graph
    
    def depth_first_search(self, start: int, goal: int) -> SearchResult:
        """
        Depth-First Search implementation.
        Using stack for frontier management.
        """
        if start == goal:
            return SearchResult([start], 0, 1, "DFS")
        
        stack = [(start, [start], 0)]  # (current_building, path, total_time)
        visited = set()
        nodes_expanded = 0
        
        while stack:
            current, path, total_time = stack.pop()
            nodes_expanded += 1
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "DFS")
            
            # Add neighbors to stack (reverse order to maintain left-to-right exploration)
            neighbors = self.graph.get_neighbors(current)
            for neighbor_id, travel_time in reversed(neighbors):
                if neighbor_id not in visited:
                    stack.append((neighbor_id, path + [neighbor_id], total_time + travel_time))
        
        return SearchResult([], 0, nodes_expanded, "DFS")  # No path found
    
    def breadth_first_search(self, start: int, goal: int) -> SearchResult:
        """
        bfs implementation.
        Uses queue for frontier management.
        """
        if start == goal:
            return SearchResult([start], 0, 1, "BFS")
        
        queue = deque([(start, [start], 0)])  # (current_building, path, total_time)
        visited = set()
        nodes_expanded = 0
        
        while queue:
            current, path, total_time = queue.popleft()
            nodes_expanded += 1
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "BFS")
            
            # Add neighbors to queue
            neighbors = self.graph.get_neighbors(current)
            for neighbor_id, travel_time in neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id], total_time + travel_time))
        
        return SearchResult([], 0, nodes_expanded, "BFS")  # No path found
    
    def iterative_deepening_search(self, start: int, goal: int) -> SearchResult:
        """
        Iterative Deepening Search implementation.
        Combines BFS optimality with DFS space efficiency.
        """
        if start == goal:
            return SearchResult([start], 0, 1, "IDS")
        
        depth = 0
        total_nodes_expanded = 0
        
        while True:
            result = self._depth_limited_search(start, goal, depth)
            total_nodes_expanded += result.nodes_expanded
            
            if result.path:  # Path found
                result.nodes_expanded = total_nodes_expanded
                result.algorithm_name = "IDS"
                return result
            
            if not result.path and result.nodes_expanded == 0:
                # No more nodes to explore at this depth
                break
            
            depth += 1
        
        return SearchResult([], 0, total_nodes_expanded, "IDS")
    
    def _depth_limited_search(self, start: int, goal: int, limit: int) -> SearchResult:
        """Helper function for IDS - performs depth-limited search."""
        if start == goal:
            return SearchResult([start], 0, 1, "DLS")
        
        stack = [(start, [start], 0, 0)]  # (current_building, path, total_time, depth)
        nodes_expanded = 0
        
        while stack:
            current, path, total_time, depth = stack.pop()
            nodes_expanded += 1
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "DLS")
            
            if depth < limit:
                # Add neighbors to stack
                neighbors = self.graph.get_neighbors(current)
                for neighbor_id, travel_time in reversed(neighbors):
                    if neighbor_id not in path:  # Avoid cycles
                        stack.append((neighbor_id, path + [neighbor_id], total_time + travel_time, depth + 1))
        
        return SearchResult([], 0, nodes_expanded, "DLS")
    
    def uniform_cost_search(self, start: int, goal: int) -> SearchResult:
        """
        Uniform Cost Search implementation.
        Uses priority queue ordered by path cost (total travel time).
        """
        if start == goal:
            return SearchResult([start], 0, 1, "UCS")
        
        # Priority queue: (total_cost, current_building, path, total_time)
        frontier = [(0, start, [start], 0)]
        visited = set()
        nodes_expanded = 0
        
        while frontier:
            cost, current, path, total_time = heapq.heappop(frontier)
            nodes_expanded += 1
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "UCS")
            
            # Add neighbors to frontier
            neighbors = self.graph.get_neighbors(current)
            for neighbor_id, travel_time in neighbors:
                if neighbor_id not in visited:
                    new_cost = total_time + travel_time
                    new_path = path + [neighbor_id]
                    heapq.heappush(frontier, (new_cost, neighbor_id, new_path, new_cost))
        
        return SearchResult([], 0, nodes_expanded, "UCS")  # No path found
    
    def greedy_best_first_search(self, start: int, goal: int) -> SearchResult:
        """
        Greedy Best-First Search implementation.
        Uses heuristic function to guide search toward goal.
        """
        if start == goal:
            return SearchResult([start], 0, 1, "Greedy BFS")
        
        # Priority queue: (heuristic_cost, current_building, path, total_time)
        frontier = [(self.graph.heuristic_distance(start, goal), start, [start], 0)]
        visited = set()
        nodes_expanded = 0
        
        while frontier:
            heuristic_cost, current, path, total_time = heapq.heappop(frontier)
            nodes_expanded += 1
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "Greedy BFS")
            
            # Add neighbors to frontier
            neighbors = self.graph.get_neighbors(current)
            for neighbor_id, travel_time in neighbors:
                if neighbor_id not in visited:
                    heuristic = self.graph.heuristic_distance(neighbor_id, goal)
                    new_path = path + [neighbor_id]
                    new_time = total_time + travel_time
                    heapq.heappush(frontier, (heuristic, neighbor_id, new_path, new_time))
        
        return SearchResult([], 0, nodes_expanded, "Greedy BFS")  # No path found
    
    def a_star_search(self, start: int, goal: int) -> SearchResult:
        """
        A* Search implementation.
        Uses f(n) = g(n) + h(n) where g(n) is path cost and h(n) is heuristic.
        """
        if start == goal:
            return SearchResult([start], 0, 1, "A*")
        
        # Priority queue: (f_cost, current_building, path, total_time)
        frontier = [(self.graph.heuristic_distance(start, goal), start, [start], 0)]
        visited = set()
        nodes_expanded = 0
        
        while frontier:
            f_cost, current, path, total_time = heapq.heappop(frontier)
            nodes_expanded += 1
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return SearchResult(path, total_time, nodes_expanded, "A*")
            
            # Add neighbors to frontier
            neighbors = self.graph.get_neighbors(current)
            for neighbor_id, travel_time in neighbors:
                if neighbor_id not in visited:
                    new_time = total_time + travel_time
                    heuristic = self.graph.heuristic_distance(neighbor_id, goal)
                    f_cost = new_time + heuristic
                    new_path = path + [neighbor_id]
                    heapq.heappush(frontier, (f_cost, neighbor_id, new_path, new_time))
        
        return SearchResult([], 0, nodes_expanded, "A*")  # No path found


def load_test_routes(filename: str) -> List[Tuple[int, int]]:
    """Load test route requests from file."""
    routes = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                routes.append((int(parts[0]), int(parts[1])))
    return routes


def main():
    """Main function to run the campus navigation system."""
    print("Smart Campus Navigation System")
    print("=" * 50)
    
    # Load campus data
    campus = CampusGraph()
    campus.load_from_files('buildings.csv', 'walkways.csv')
    
    print(f"Loaded {len(campus.buildings)} buildings:")
    for building in campus.buildings.values():
        print(f"  {building}")
    
    print(f"\nLoaded {len(campus.adjacency_list)} building connections")
    
    # Load test routes
    test_routes = load_test_routes('test_routes.txt')
    print(f"\nTest routes to evaluate:")
    for i, (start, goal) in enumerate(test_routes, 1):
        start_name = campus.buildings[start].name
        goal_name = campus.buildings[goal].name
        print(f"  Route {i}: {start_name} ({start}) -> {goal_name} ({goal})")
    
    # Initialize navigator
    navigator = CampusNavigator(campus)
    
    # Define search algorithms
    algorithms = [
        ("DFS", navigator.depth_first_search),
        ("BFS", navigator.breadth_first_search),
        ("IDS", navigator.iterative_deepening_search),
        ("UCS", navigator.uniform_cost_search),
        ("Greedy BFS", navigator.greedy_best_first_search),
        ("A*", navigator.a_star_search)
    ]
    
    # Results table
    print("\n" + "=" * 80)
    print("SEARCH ALGORITHM COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Route':<15} {'Algorithm':<12} {'Travel Time':<12} {'Nodes Expanded':<15} {'Path'}")
    print("-" * 80)
    
    results = []
    
    for route_num, (start, goal) in enumerate(test_routes, 1):
        start_name = campus.buildings[start].name
        goal_name = campus.buildings[goal].name
        route_name = f"Route {route_num}"
        
        for algorithm_name, algorithm_func in algorithms:
            result = algorithm_func(start, goal)
            results.append((route_num, algorithm_name, result))
            
            # Format path for display
            path_str = " -> ".join([campus.buildings[building_id].name for building_id in result.path])
            
            print(f"{route_name:<15} {algorithm_name:<12} {result.total_time:<12} {result.nodes_expanded:<15} {path_str}")
        


if __name__ == "__main__":
    main()
