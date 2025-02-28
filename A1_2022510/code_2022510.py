import numpy as np
import pickle
import json
import heapq
import math
from collections import deque

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_ids_path(adj_matrix, start_node, goal_node):
    def dfs_limited(adj_matrix, node, goal, depth, path, visited):
        if node == goal:
            return path + [node]
        if depth == 0:
            return None
        visited.add(node)
        for neighbor, is_edge in enumerate(adj_matrix[node]):
            if is_edge and neighbor not in visited:
                result = dfs_limited(adj_matrix, neighbor, goal, depth - 1, path + [node], visited)
                if result:
                    return result
        # visited.remove(node)  #since it is a graph not a tree
        return None

    def ids(adj_matrix, start, goal, max_depth):
        for depth in range(max_depth):
            visited = set()
            path = dfs_limited(adj_matrix, start, goal, depth, [], visited)
            if path:
                return path
        return []

    max_depth = len(adj_matrix)
    return ids(adj_matrix, start_node, goal_node, max_depth)


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]


def get_bidirectional_search_path(adj_matrix, start_node, goal_node):    
    # Edge case
    if start_node == goal_node:
        return [start_node]

    # BFS function for both start or goal traversal
    def bfs_directional(frontier, visited_from_this_side, visited_from_other_side, parents_from_this_side):
        level_size = len(frontier)
        for _ in range(level_size):
            current_node = frontier.popleft()
            
            for neighbor, is_edge in enumerate(adj_matrix[current_node]):
                if is_edge and neighbor not in visited_from_this_side:
                    visited_from_this_side.add(neighbor)
                    frontier.append(neighbor)
                    parents_from_this_side[neighbor] = current_node
                    if neighbor in visited_from_other_side:
                        return neighbor
        return None
    
    # Function to get the path using parent
    def reconstruct_path(parents_from_start, parents_from_goal, meeting_node):       
        path = []
        curr = meeting_node
        while curr is not None:
            path.append(curr)
            curr = parents_from_start[curr]
        path.reverse()

        curr = parents_from_goal[meeting_node]
        while curr is not None:
            path.append(curr)
            curr = parents_from_goal[curr]

        return path
    
    # Main code begins
    start_frontier = deque([start_node])
    goal_frontier = deque([goal_node])
    
    visited_from_start = {start_node}
    visited_from_goal = {goal_node}
    
    parents_from_start = {start_node: None}
    parents_from_goal = {goal_node: None}

    while start_frontier and goal_frontier:
        meeting_node = bfs_directional(start_frontier, visited_from_start, visited_from_goal, parents_from_start)
        if meeting_node is not None:
            return reconstruct_path(parents_from_start, parents_from_goal, meeting_node)

        meeting_node = bfs_directional(goal_frontier, visited_from_goal, visited_from_start, parents_from_goal)
        if meeting_node is not None:
            return reconstruct_path(parents_from_start, parents_from_goal, meeting_node)

    return []


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 9, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def euclidean_distance(a, b):
        x1, y1 = node_attributes[a]['x'], node_attributes[a]['y']
        x2, y2 = node_attributes[b]['x'], node_attributes[b]['y']
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def heuristic(node):
        return euclidean_distance(start_node, node) + euclidean_distance(node, goal_node)
    
    def reconstruct_path(parent, curr):
        path = []
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path

    # Main code begins
    frontier = []
    g_score_start = {start_node: 0}
    parent = {start_node: None}
    heapq.heappush(frontier, (g_score_start[start_node] + heuristic(start_node), start_node))
    
    while frontier:
        _, current = heapq.heappop(frontier)
    
        if current == goal_node:
            return reconstruct_path(parent, goal_node)
        
        for neighbor, is_edge in enumerate(adj_matrix[current]):
            if is_edge:
                # g_new = g_score_start[current] + euclidean_distance(current, neighbor)
                g_new = g_score_start[current] + adj_matrix[current][neighbor]
                
                if neighbor not in g_score_start or g_new < g_score_start[neighbor]:
                    g_score_start[neighbor] = g_new
                    f_score = g_new + heuristic(neighbor)
                    heapq.heappush(frontier, (f_score, neighbor))
                    parent[neighbor] = current
    
    return []


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]


def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    
    # Function to calculate Euclidean distance b/w two points 
    def euclidean_distance(a, b):
        x1, y1 = node_attributes[a]['x'], node_attributes[a]['y']
        x2, y2 = node_attributes[b]['x'], node_attributes[b]['y']
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Function to calculate heuuristic h(w) = dist(u,w) + dist(v,w)
    def heuristic(node, start, goal):
        return euclidean_distance(start, node) + euclidean_distance(node, goal)
    
    # A star function to search both sides once at a time
    def a_star_directional(frontier, visited_this_side, visited_other_side, g_scores_this_side, new_start, new_goal, is_forward):
        _, current = heapq.heappop(frontier)

        if current in visited_other_side:
            return reconstruct_path(visited_this_side, visited_other_side, current, is_forward)
        
        for neighbor, is_edge in enumerate(adj_matrix[current]):
            if is_edge:
                # g_new = g_scores_this_side[current] + euclidean_distance(current, neighbor)
                g_new = g_scores_this_side[current] + adj_matrix[current][neighbor]
                
                if neighbor not in g_scores_this_side or g_new < g_scores_this_side[neighbor]:
                    g_scores_this_side[neighbor] = g_new
                    f_score = g_new + heuristic(neighbor, new_start, new_goal)
                    heapq.heappush(frontier, (f_score, neighbor))
                    visited_this_side[neighbor] = current
                    
        return None

    def reconstruct_path(visited_from_start, visited_from_goal, meeting_node, is_forward):
        path_start = []
        curr = meeting_node
        while curr is not None:
            path_start.append(curr)
            curr = visited_from_start[curr]
        
        path_goal = []
        curr = visited_from_goal[meeting_node]
        while curr is not None:
            path_goal.append(curr)
            curr = visited_from_goal[curr]
        
        if is_forward:
            return path_start[::-1] + path_goal
        else:
            return path_goal[::-1] + path_start


    # Main code begins
    frontier_start = []
    frontier_goal = []

    g_score_start = {start_node: 0}
    g_score_goal = {goal_node: 0}

    heapq.heappush(frontier_start, (heuristic(start_node, start_node, goal_node), start_node))
    heapq.heappush(frontier_goal, (heuristic(goal_node, goal_node, start_node), goal_node))

    visited_from_start = {start_node: None}
    visited_from_goal = {goal_node: None}

    while frontier_start and frontier_goal:
        path = a_star_directional(frontier_start, visited_from_start, visited_from_goal, g_score_start, start_node, goal_node, True)
        if path:
            return path
        
        path = a_star_directional(frontier_goal, visited_from_goal, visited_from_start, g_score_goal, goal_node, start_node, False)
        if path:
            return path
        
    return []


# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    
    tin = [-1]*n
    mn = [-1]*n
    parent = [-1]*n
    timer = [0]
    
    res = []
    def dfs(u):
        tin[u] = mn[u] = timer[0]
        timer[0] += 1

        for v in range(n):
            if adj_matrix[u][v]:
                if tin[v] == -1:
                    parent[v] = u
                    dfs(v)
                    
                    mn[u] = min(mn[u], mn[v])
                    if mn[v] > tin[u]:
                        res.append((u, v))
                        
                elif tin[v] != -1 and v != parent[u]:
                    mn[u] = min(mn[u], tin[v])

    for i in range(n):
        if tin[i] == -1:
            dfs(i)

    return res


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')