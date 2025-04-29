import random
import numpy as np
import config
import concurrent.futures
import os
import pickle
import time
from models.ev import EV
from models.station import ChargingStation
from models.maps_service import get_route, calculate_distance, save_cache

BANGALORE_CENTER = (12.9716, 77.5946)

CITY_RADIUS = 0.1  # ~11km

# Cache for generated data
NODES_CACHE_FILE = "nodes_cache.pkl"
ROUTES_CACHE_FILE = "routes_cache.pkl"

def generate_random_location(center=BANGALORE_CENTER, radius=CITY_RADIUS):
    """Generate a random location within radius of center"""
    angle = random.uniform(0, 2 * np.pi)
    
    distance = random.uniform(0, radius)
    
    # Convert to lat/lng offset
    lat_offset = distance * np.cos(angle)
    lng_offset = distance * np.sin(angle)
    
    lat = center[0] + lat_offset
    lng = center[1] + lng_offset
    
    return (lat, lng)

def save_nodes(nodes):
    try:
        with open(NODES_CACHE_FILE, 'wb') as f:
            pickle.dump(nodes, f)
    except Exception as e:
        print(f"Error saving nodes cache: {e}")

def load_nodes():
    if os.path.exists(NODES_CACHE_FILE):
        try:
            with open(NODES_CACHE_FILE, 'rb') as f:
                nodes = pickle.load(f)
            print(f"Loaded {len(nodes)} cached nodes")
            return nodes
        except Exception as e:
            print(f"Error loading nodes cache: {e}")
    return None

def save_routes(routes):
    try:
        with open(ROUTES_CACHE_FILE, 'wb') as f:
            pickle.dump(routes, f)
    except Exception as e:
        print(f"Error saving routes cache: {e}")

def load_routes():
    if os.path.exists(ROUTES_CACHE_FILE):
        try:
            with open(ROUTES_CACHE_FILE, 'rb') as f:
                routes = pickle.load(f)
            print(f"Loaded {len(routes)} cached routes")
            return routes
        except Exception as e:
            print(f"Error loading routes cache: {e}")
    return None

def interpolate_route(origin, destination, segment_km=0.5):
    """Interpolate straight-line route into ~segment_km segments."""
    total_m = calculate_distance(origin, destination)
    total_km = total_m / 1000.0
    if total_km <= segment_km:
        return [origin, destination]
    num_segments = max(2, int(total_km / segment_km) + 1)
    points = []
    for i in range(num_segments):
        t = i / (num_segments - 1)
        lat = origin[0] + t * (destination[0] - origin[0])
        lng = origin[1] + t * (destination[1] - origin[1])
        points.append((lat, lng))
    return points


def generate_synthetic_data(num_evs=100, num_stations=20, num_nodes=80, num_routes=240, use_cache=True, seed=None):
    """
    Generate synthetic EVs, charging stations, nodes and routes
    
    Args:
        num_evs: Number of EVs to generate
        num_stations: Number of charging stations to generate
        num_nodes: Number of nodes (locations) in the city
        num_routes: Number of predefined routes between nodes
        use_cache: Whether to use cached data if available
        seed: Optional random seed for reproducible generation
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Try to load nodes from cache
    nodes = None
    routes = None
    if use_cache:
        nodes = load_nodes()
        routes = load_routes()
    
    if nodes is None:
        print("Generating nodes...")
        nodes = [generate_random_location() for _ in range(num_nodes)]
        save_nodes(nodes)
    else:
        # Ensure we have enough nodes
        if len(nodes) < num_nodes:
            print(f"Extending nodes from {len(nodes)} to {num_nodes}...")
            additional_nodes = [generate_random_location() for _ in range(num_nodes - len(nodes))]
            nodes.extend(additional_nodes)
            save_nodes(nodes)
    
    print("Generating charging stations...")
    stations = []
    for i in range(num_stations):
        location = random.choice(nodes)
        num_chargers = random.randint(1, 4)
        charging_rate = random.choice([7.0, 11.0, 22.0])
        
        station = ChargingStation(
            id=f"station-{i+1}",
            location=location,
            num_chargers=num_chargers,
            charging_rate=charging_rate
        )
        stations.append(station)
    
    # Generate predefined routes between nodes if not loaded from cache
    if routes is None or len(routes) < num_routes:
        print(f"Generating routes in parallel (this might take a while)...")
        routes = generate_routes_parallel(nodes, num_routes, existing_routes=routes)
        routes.sort(key=lambda r: r.get("id", ""))
        # Save routes to cache after generation
        save_routes(routes)
    else:
        print(f"Using {len(routes)} cached routes")
    
    print("Generating EVs...")
    evs = []
    for i in range(num_evs):
        selected_route = random.choice(routes)
        
        # Generate EV properties
        # 20-60 kWh for battery capacity (smaller for scooters, larger for cars)
        battery_capacity = random.uniform(20, 60)
        # 0.2 to 0.8 for initial SoC
        initial_soc = random.uniform(0.2, 0.8)
        # 0.15 to 0.25 kWh/km for consumption
        consumption_rate = random.uniform(0.15, 0.25)
        
        ev = EV(
            id=f"ev-{i+1}",
            origin=selected_route["origin"],
            destination=selected_route["destination"],
            battery_capacity=battery_capacity,
            initial_soc=initial_soc,
            consumption_rate=consumption_rate,
            route=selected_route["points"]
        )
        evs.append(ev)
    
    save_cache()
    print("Data generation complete!")
    return evs, stations, routes

def generate_routes_parallel(nodes, num_routes, max_workers=10, existing_routes=None):
    """Generate routes in parallel using ThreadPoolExecutor"""
    routes = [] if existing_routes is None else existing_routes.copy()
    routes_to_generate = max(0, num_routes - len(routes))
    
    if routes_to_generate == 0:
        print("No new routes to generate")
        return routes
    
    print(f"Generating {routes_to_generate} new routes...")
    
    # Create node pairs for routes
    node_pairs = []
    for _ in range(routes_to_generate):
        # Select random origin and destination nodes
        origin_idx = random.randint(0, len(nodes) - 1)
        dest_idx = random.randint(0, len(nodes) - 1)
        
        # Ensure origin and destination are different
        while origin_idx == dest_idx:
            dest_idx = random.randint(0, len(nodes) - 1)
            
        node_pairs.append((nodes[origin_idx], nodes[dest_idx]))
    
    # Process routes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pair = {
            executor.submit(get_route_data, pair[0], pair[1], len(routes) + i): pair 
            for i, pair in enumerate(node_pairs)
        }
        
        # Process completed futures with timeout handling
        completed = 0
        total = len(node_pairs)
        
        try:
            completed_routes = []
            for future in concurrent.futures.as_completed(future_to_pair, timeout=60):
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"Generated {completed}/{total} routes...")
                    # Save interim progress
                    if completed % 50 == 0 and routes:
                        save_routes(routes)

                try:
                    route_data, route_id = future.result(timeout=30)
                    completed_routes.append((route_id, route_data))
                except concurrent.futures.TimeoutError:
                    print(f"Route generation timed out for pair {future_to_pair[future]}")
                except Exception as e:
                    print(f"Error generating route: {e}")

            completed_routes.sort(key=lambda x: x[0])
            for _, route_data in completed_routes:
                routes.append(route_data)
        
        except concurrent.futures.TimeoutError:
            print("Route generation taking too long, saving partial results...")
        except KeyboardInterrupt:
            print("Operation interrupted by user, saving partial results...")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            # Shutdown executor (non-blocking)
            executor.shutdown(wait=False)
            
            # Save whatever routes we have
            if routes:
                print(f"Saving {len(routes)} routes generated so far")
                save_routes(routes)
    
    return routes

def get_route_data(origin, destination, route_id):
    """Get route data between two points with error handling"""
    try:
        route_data = get_route(origin, destination)
        distance = route_data["distance"] / 1000  # Convert m to km
        
        result = {
            "id": f"route-{route_id+1}",
            "origin": origin,
            "destination": destination,
            "points": route_data["points"],
            "distance": distance
        }
        return result, route_id
    except Exception as e:
        print(f"API error for route {route_id}: {str(e)[:100]}")
        # Fallback if route API fails — interpolate ~0.5 km segments
        route = interpolate_route(origin, destination, segment_km=0.5)
        # Estimate distance (straight line)
        distance = calculate_distance(origin, destination) / 1000  # Convert m to km
        
        result = {
            "id": f"route-{route_id+1}",
            "origin": origin,
            "destination": destination,
            "points": route,
            "distance": distance
        }
        return result, route_id