import requests
import math
import config
import os
import json
import pickle

# File-based persistent cache
CACHE_FILE = "route_cache.pkl"
_route_cache = {}

# Load cache from disk if it exists
def load_cache():
    global _route_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                _route_cache = pickle.load(f)
            print(f"Loaded {len(_route_cache)} cached routes")
    except Exception as e:
        print(f"Error loading route cache: {e}")
        _route_cache = {}

# Save cache to disk
def save_cache():
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(_route_cache, f)
    except Exception as e:
        print(f"Error saving route cache: {e}")

# Load cache at module import time
load_cache()

def get_route(origin, destination):
    """
    Get route between origin and destination using Google Maps Directions API
    
    Args:
        origin (tuple): (lat, lng) of origin
        destination (tuple): (lat, lng) of destination
        
    Returns:
        dict: Contains route details including points, distance and duration
    """
    # Create a cache key from origin and destination
    cache_key = f"{origin[0]},{origin[1]}-{destination[0]},{destination[1]}"
    
    # Check cache first
    if cache_key in _route_cache:
        print(f"Cache hit for route: {cache_key[:20]}...")
        return _route_cache[cache_key]
        
    # Format coordinates for API
    origin_str = f"{origin[0]},{origin[1]}"
    destination_str = f"{destination[0]},{destination[1]}"
    
    # API endpoint
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Parameters
    params = {
        "origin": origin_str,
        "destination": destination_str,
        "key": config.GOOGLE_MAPS_API_KEY
    }
    
    # Make request
    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if request was successful
    if data["status"] != "OK":
        raise Exception(f"Error fetching directions: {data['status']}")
    
    # Extract route details
    route = data["routes"][0]
    legs = route["legs"][0]
    
    # Get steps and decode to list of points
    points = []
    for step in legs["steps"]:
        points.append((step["start_location"]["lat"], step["start_location"]["lng"]))
    
    # Add final point
    points.append((legs["end_location"]["lat"], legs["end_location"]["lng"]))
    
    # Get distance and duration
    distance = legs["distance"]["value"]  # meters
    duration = legs["duration"]["value"]  # seconds
    
    result = {
        "points": points,
        "distance": distance,
        "duration": duration
    }
    
    # Store in cache before returning
    _route_cache[cache_key] = result
    
    # Save to disk every 10 new routes
    if len(_route_cache) % 10 == 0:
        save_cache()
    
    return result

def calculate_distance(origin, destination):
    """Calculate straight-line distance between two points (fallback)"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lat1, lon1 = origin
    lat2, lon2 = destination
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    
    return c * r