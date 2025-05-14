import time
import logging
import traceback
from models.maps_service import calculate_distance

# Set up logger
optimization_logger = logging.getLogger("optimization")
optimization_logger.setLevel(logging.INFO)

# In-memory handler for log messages
class InMemoryHandler(logging.Handler):
    def __init__(self, max_entries=100):
        super().__init__()
        self.log_entries = []
        self.max_entries = max_entries
        
    def emit(self, record):
        if len(self.log_entries) >= self.max_entries:
            self.log_entries.pop(0)
        self.log_entries.append(self.format(record))
        
    def get_logs(self):
        return self.log_entries

# Create and add handler
log_handler = InMemoryHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
optimization_logger.addHandler(log_handler)

def get_optimization_logs():
    """Get the current optimization logs"""
    return log_handler.get_logs()

def optimize_charging(evs, stations):
    """
    Smart charging station assignment based on accessibility, wait time, and energy needs
    
    Args:
        evs (list): List of EVs needing charging
        stations (list): List of available charging stations
        
    Returns:
        dict: Mapping of EV IDs to assigned station IDs
        list: List of EV IDs that could not be assigned (emergency)
    """
    if not evs or not stations:
        optimization_logger.info(f"No optimization needed: EVs={len(evs)}, Stations={len(stations)}")
        return {}, []
        
    start_time = time.time()
    optimization_logger.info(f"Starting optimization for {len(evs)} EVs and {len(stations)} stations")
    
    assignments = {}
    abandoned_evs = []
    
    try:
        for ev in evs:
            try:
                optimization_logger.info(f"Optimizing for EV {ev.id} - Current SoC: {ev.soc:.2f}")
                
                # Safety check for invalid route data
                if not ev.route or len(ev.route) < 2:
                    optimization_logger.warning(f"EV {ev.id} has invalid route data, skipping")
                    continue
                    
                # Safety check for route index out of bounds
                if ev.route_index >= len(ev.route):
                    optimization_logger.warning(f"EV {ev.id} has invalid route index {ev.route_index}, resetting to 0")
                    ev.route_index = 0
                
                # Find stations that are reachable with current battery
                reachable_stations = []
                for station in stations:
                    try:
                        if ev.can_reach_station(station.location):
                            # Check if station is on or near route (within 1000m detour)
                            on_route = ev.is_station_on_route(station.location, max_detour=1000)
                            reachable_stations.append({
                                'station': station,
                                'on_route': on_route
                            })
                    except Exception as station_error:
                        optimization_logger.error(f"Error checking station {station.id} for EV {ev.id}: {station_error}")
                        continue
                
                optimization_logger.info(f"Found {len(reachable_stations)} reachable stations for EV {ev.id}")
                
                if not reachable_stations:
                    optimization_logger.warning(f"No reachable stations for EV {ev.id}! Marking as abandoned.")
                    abandoned_evs.append(ev.id)
                    continue
                
                # Sort by preference: first on-route stations, then others
                reachable_stations.sort(key=lambda x: 0 if x['on_route'] else 1)
                
                # Calculate scores for each reachable station
                station_scores = []
                for station_data in reachable_stations:
                    try:
                        station = station_data['station']
                        on_route = station_data['on_route']
                        
                        # Calculate travel distance and time
                        travel_distance = calculate_distance(ev.current_position, station.location) / 1000  # km
                        travel_time = travel_distance / 30 * 3600  # seconds, assuming 30 km/h
                        
                        # Get estimated wait time at this station
                        wait_time = station.get_current_wait_time_estimate()
                        
                        # Calculate charging time needed
                        # Assume we need to charge to reach destination + 10km buffer
                        energy_needed = ev.calculate_energy_needed_for_destination()
                        current_energy = ev.soc * ev.battery_capacity
                        energy_to_charge = max(0, energy_needed - current_energy)
                        
                        # Ensure we always charge at least 10% of battery capacity
                        min_charge = 0.1 * ev.battery_capacity
                        energy_to_charge = max(energy_to_charge, min_charge)
                        
                        charge_time = (energy_to_charge / station.charging_rate) * 3600  # seconds
                        
                        # Adjust charge time based on queue length
                        # Cannot charge more than 80% if queue exists, unless energy needed is greater
                        if station.get_queue_length() > 0 or len(station.charging_evs) >= station.num_chargers:
                            # Calculate energy for 80% SoC
                            energy_80pct = 0.8 * ev.battery_capacity - current_energy
                            # If energy needed is greater than what 80% provides, use energy needed
                            # Otherwise, use 80% cap
                            if energy_needed > 0.8 * ev.battery_capacity:
                                energy_to_charge = max(energy_to_charge, min_charge)
                            else:
                                energy_to_charge = min(energy_to_charge, max(energy_80pct, min_charge))
                            
                            charge_time = (energy_to_charge / station.charging_rate) * 3600  # seconds
                        
                        # Total time = travel time + wait time + charge time
                        total_time = travel_time + wait_time + charge_time
                        
                        # Apply bonus for on-route stations (reduce score)
                        if on_route:
                            total_time *= 0.9  # 10% bonus for on-route stations
                        
                        # Critical battery penalty (< 10% battery)
                        if ev.soc < 0.1:
                            # If battery very low, heavily penalize far stations
                            total_time += travel_time * 5
                        
                        station_scores.append({
                            'station_id': station.id,
                            'travel_time': travel_time,
                            'wait_time': wait_time,
                            'charge_time': charge_time,
                            'total_time': total_time,
                            'on_route': on_route
                        })
                        
                        optimization_logger.info(f"Station {station.id} score: travel={travel_time:.1f}s, "
                                    f"wait={wait_time:.1f}s, charge={charge_time:.1f}s, "
                                    f"total={total_time:.1f}s, on_route={on_route}")
                    except Exception as score_error:
                        optimization_logger.error(f"Error calculating score for station {station_data['station'].id}: {score_error}")
                        continue
                
                # Choose station with lowest total time
                if station_scores:
                    best_station = min(station_scores, key=lambda x: x['total_time'])
                    assignments[ev.id] = best_station['station_id']
                    
                    optimization_logger.info(f"Assigned EV {ev.id} to station {best_station['station_id']} "
                                f"with total time: {best_station['total_time']:.1f}s")
                else:
                    # No valid scores - mark as abandoned
                    optimization_logger.warning(f"No valid station scores for EV {ev.id}! Marking as abandoned.")
                    abandoned_evs.append(ev.id)
            except Exception as ev_error:
                optimization_logger.error(f"Error optimizing for EV {ev.id}: {ev_error}")
                optimization_logger.error(traceback.format_exc())
                # Mark as abandoned due to error
                abandoned_evs.append(ev.id)
                continue
    except Exception as e:
        optimization_logger.error(f"Critical optimization error: {e}")
        optimization_logger.error(traceback.format_exc())
    
    optimization_time = time.time() - start_time
    optimization_logger.info(f"Optimization completed in {optimization_time:.3f} seconds")
    optimization_logger.info(f"Assigned {len(assignments)} EVs out of {len(evs)}")
    
    if abandoned_evs:
        optimization_logger.warning(f"{len(abandoned_evs)} EVs abandoned due to unsolvable situations: {abandoned_evs}")
    
    return assignments, abandoned_evs