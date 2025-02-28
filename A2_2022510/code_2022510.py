# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# df_stops = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\stops.txt')
# df_routes = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\routes.txt')
# df_stop_times = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\stop_times.txt')
# df_fare_attributes = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\fare_attributes.txt')
# df_trips = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\trips.txt')
# df_fare_rules = pd.read_csv(r'C:\Users\Subham Maurya\OneDrive\Documents\Artificial Intelligence\Assignment2\GTFS\fare_rules.txt')
# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data   
def create_kb():
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    routes_and_stops_sorted = df_stop_times.groupby('trip_id').apply(lambda x:   x.sort_values('stop_sequence')['stop_id'].tolist())
    for trip_id,stops in routes_and_stops_sorted.items():
        route_id = trip_to_route.get(trip_id)      # assuming that there is a route_id for every trip_id in trip_to_route
        route_to_stops[route_id].extend(stops)

    for route_id in route_to_stops:
        route_to_stops[route_id] = list(dict.fromkeys(route_to_stops[route_id]))

    stop_trip_count.update(df_stop_times['stop_id'].value_counts().to_dict())
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='left')
    fare_rules = merged_fare_df.groupby('route_id').apply(lambda x: x.to_dict(orient='records')).to_dict()

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    count_trips_for_route = defaultdict(int)
    for _, route_id in trip_to_route.items():
        count_trips_for_route[route_id] += 1

    res = sorted(count_trips_for_route.items(), key =lambda x: x[1], reverse=True)[:5]
    return res

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    res = sorted(stop_trip_count.items() , key=lambda x:x[1], reverse= True)[:5]
    return res

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    routes_for_stops = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            routes_for_stops[stop_id].add(route_id)

    cnt = {stop_id: len(routes) for stop_id, routes in routes_for_stops.items()}
    res = sorted(cnt.items() , key= lambda x:x[1] , reverse =True)[:5]
    return res

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    unique_pair_of_stops = defaultdict(list)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_1, stop_2 = stops[i], stops[i + 1]
            unique_pair_of_stops[(stop_1, stop_2)].append(route_id)

    temp_route_pair = []
    for (stop_1, stop_2), routes in unique_pair_of_stops.items():
        if len(routes) == 1:                # Only  one  direct  route
            route_id = routes[0]
            both = stop_trip_count[stop_1] + stop_trip_count[stop_2]
            temp_route_pair.append(((stop_1, stop_2), route_id, both))

    res = sorted(temp_route_pair, key= lambda x:x[2], reverse =True)[:5]
    return [(pair, route_id) for pair, route_id, useless in res]

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph(route_to_stops):
    G = nx.Graph()

    # Add edges for each route based on consecutive stops
    for stops in route_to_stops.values():
        G.add_edges_from(zip(stops[:-1], stops[1:]))

    pos = nx.spring_layout(G, seed=42)  #starting from random postion

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Edge trace
    edge_trace = go.Scatter(x = edge_x, y = edge_y, line = dict(width=0.6, color = '#1234CC'), hoverinfo = 'none', mode = 'lines')

    # Node trace 
    node_x, node_y = zip(*[pos[node] for node in G.nodes()])
    node_trace = go.Scatter(
        x=node_x, y=node_y, 
        mode='markers+text',
        text=[f"ID: {node}" for node in G.nodes()],
        textposition="top center",hoverinfo='text',
    )

    fig = go.Figure(data=[edge_trace, node_trace],layout=go.Layout(title='Visualization of Stops and Routes Graph',))

    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    direct_routes = []
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            if stops.index(start_stop) < stops.index(end_stop):
                direct_routes.append(route_id)

    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')
def initialize_datalog():
    pyDatalog.clear()
    add_route_data(route_to_stops)

    pyDatalog.create_terms('RouteHasStop, DirectRoute, X, Y, R')
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)

# Adding route data to Datalog
def add_route_data(route_to_stops):
    for route_id, stops in route_to_stops.items():
        for _ , stop_id in enumerate(stops):
            +RouteHasStop(route_id, stop_id)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    ans = DirectRoute(R, start, end)
    res = [row[0] for row in ans]
    return res

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id , end_stop_id, stop_id_to_include , max_transfers):
    pyDatalog.clear()
    add_route_data(route_to_stops)
    
    pyDatalog.create_terms('RouteHasStop, DirectRoute, R1, R2')
    DirectRoute(R1, stop_id_to_include, R2) <= (
        RouteHasStop(R1, start_stop_id) & RouteHasStop(R1, stop_id_to_include) & 
        RouteHasStop(R2, end_stop_id) & RouteHasStop(R2, stop_id_to_include) & 
        (R1 != R2)
    )
    
    ans = DirectRoute(R1, stop_id_to_include, R2)
    res = [(row[0], stop_id_to_include, row[1]) for row in ans if max_transfers>=1]
    return res

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    pyDatalog.clear()
    add_route_data(route_to_stops)
    
    pyDatalog.create_terms('RouteHasStop, DirectRoute, R1, R2')
    DirectRoute(R1, stop_id_to_include, R2) <= (
        RouteHasStop(R2, end_stop_id) & RouteHasStop(R2, stop_id_to_include) & 
        RouteHasStop(R1, start_stop_id) & RouteHasStop(R1, stop_id_to_include) & 
        (R1 != R2)
    )
    
    # ans = DirectRoute(R2, stop_id_to_include, R1)
    ans = DirectRoute(R1, stop_id_to_include, R2)
    res = [(row[1],stop_id_to_include, row[0]) for row in ans if max_transfers >= 1]
    return res

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass  # Implementation here

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
