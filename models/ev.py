import uuid
from datetime import datetime
from models.maps_service import calculate_distance

class EV:
    def __init__(self, id=None, origin=None, destination=None, 
                 battery_capacity=None, initial_soc=None, 
                 consumption_rate=None, route=None):
        self.id = id or str(uuid.uuid4())
        self.origin = origin  # (lat, lng)
        self.destination = destination  # (lat, lng)
        self.battery_capacity = battery_capacity  # kWh
        self.initial_soc = initial_soc  # Preserve original SoC for simulation reset
        self.soc = initial_soc  # State of Charge (0-1)
        self.consumption_rate = consumption_rate  # kWh/km
        
        self.current_position = origin  # Start at origin
        self.route = route or []  # List of points along route [(lat, lng), ...]
        self.route_index = 0  # Current position in route
        
        self.assigned_station = None  # Station assigned for charging
        self.charging = False  # Currently at a charger
        self.in_queue = False  # Waiting in a queue
        self.queue_arrival_time = None  # When EV arrived at queue
        self.charging_start_time = None  # When EV started charging
        self.waiting_time = 0  # Total time spent waiting in queue
        self.target_soc = 0.8  # Default target SoC is 80%

        # Travel-then-queue state (used when config.SIMULATE_TRAVEL_TIME is on):
        # an assigned EV travels for `travel_time_remaining` simulated seconds
        # before joining `pending_station`'s queue, instead of teleporting in.
        self.en_route_to_charger = False
        self.pending_station = None
        self.travel_time_remaining = 0.0
        # Cumulative distance (km) traveled toward assigned charging stations,
        # used for the travel-distance comparison metric regardless of whether
        # travel time is simulated.
        self.travel_to_charger_km = 0.0

        # Simulated (not wall-clock) time spent charging, used to compute the
        # "average charging completion time" metric.
        self.charging_time_elapsed = 0.0
        self.completed_charging_duration = None
        
        self.trip_completed = False
        self.abandoned = False  # Flag for abandoned EVs (can't reach any station)
        self.trip_start_time = datetime.now()
        self.trip_end_time = None
        
        # Journey log to track detailed timeline
        self.journey_log = []
        # Record initialization
        self._log_event("Initialized", {
            "origin_node": f"Node at {origin}",
            "destination_node": f"Node at {destination}",
            "battery": f"{self.soc * 100:.1f}%",
            "total_distance": self.calculate_total_route_distance(),
            "battery_required": f"{self.calculate_energy_for_total_route() / self.battery_capacity * 100:.1f}%"
        })
    
    def _log_event(self, event_type, details):
        """Add an event to the journey log"""
        self.journey_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "details": details
        })
    
    def move(self, time_step_seconds):
        """Move the EV along its route for one time step"""
        if (self.trip_completed or self.charging or self.in_queue
                or self.en_route_to_charger or self.abandoned):
            return
        
        if self.route_index >= len(self.route) - 1:
            # Reached destination
            self.current_position = self.destination
            self.trip_completed = True
            self.trip_end_time = datetime.now()
            self._log_event("Trip Completed", {
                "final_battery": f"{self.soc * 100:.1f}%",
                "total_time": f"{(self.trip_end_time - self.trip_start_time).total_seconds()} seconds"
            })
            return
        
        # Calculate distance to next point
        current_point = self.route[self.route_index]
        next_point = self.route[self.route_index + 1]
        segment_distance = self._calculate_distance(current_point, next_point)
        
        # Calculate energy required
        energy_required = segment_distance * self.consumption_rate
        
        # Check if enough battery to complete segment
        if self.soc * self.battery_capacity >= energy_required:
            # Move to next point
            old_point = self.route[self.route_index]
            self.route_index += 1
            self.current_position = self.route[self.route_index]
            
            # Update battery
            old_soc = self.soc
            self.soc -= energy_required / self.battery_capacity
            
            # Log the movement
            self._log_event("Moved", {
                "from": f"Node at {old_point}",
                "to": f"Node at {self.current_position}",
                "distance": f"{segment_distance:.2f} km",
                "battery_used": f"{energy_required / self.battery_capacity * 100:.1f}%",
                "battery_before": f"{old_soc * 100:.1f}%",
                "battery_after": f"{self.soc * 100:.1f}%"
            })
        else:
            # Cannot complete segment
            self._log_event("Insufficient Battery", {
                "current_position": f"Node at {current_point}",
                "next_position": f"Node at {next_point}",
                "distance": f"{segment_distance:.2f} km",
                "battery_available": f"{self.soc * 100:.1f}%",
                "battery_needed": f"{energy_required / self.battery_capacity * 100:.1f}%"
            })
    
    def needs_charging(self, threshold):
        """Check if EV needs charging based on current battery and next segment needs"""
        if (self.charging or self.in_queue or self.en_route_to_charger
                or self.trip_completed or self.abandoned):
            return False
        
        # If battery below threshold, definitely need charging
        if self.soc <= threshold:
            self._log_event("Charging Needed", {
                "reason": "Battery below threshold",
                "current_battery": f"{self.soc * 100:.1f}%",
                "threshold": f"{threshold * 100:.1f}%"
            })
            return True
        
        # Check if enough battery to reach next point
        if self.route_index < len(self.route) - 1:
            current_point = self.route[self.route_index]
            next_point = self.route[self.route_index + 1]
            segment_distance = self._calculate_distance(current_point, next_point)
            energy_required = segment_distance * self.consumption_rate
            
            # Add 10% reserve requirement
            energy_with_reserve = energy_required * 1.1
            
            if self.soc * self.battery_capacity < energy_with_reserve:
                self._log_event("Charging Needed", {
                    "reason": "Insufficient battery for next segment with reserve",
                    "current_battery": f"{self.soc * 100:.1f}%",
                    "segment_distance": f"{segment_distance:.2f} km",
                    "energy_required": f"{energy_required:.2f} kWh",
                    "with_reserve": f"{energy_with_reserve:.2f} kWh"
                })
                return True
                
        return False
    
    def start_charging(self, station):
        """Start charging at a station"""
        self.assigned_station = station
        self.charging = True
        self.in_queue = False
        self.charging_start_time = datetime.now()
        
        # Log charging start
        self._log_event("Started Charging", {
            "station_id": station.id,
            "location": f"({station.location[0]:.6f}, {station.location[1]:.6f})",
            "battery_before": f"{self.soc * 100:.1f}%",
            "charging_rate": f"{station.charging_rate} kW",
            "waiting_time": f"{self.waiting_time} seconds"
        })
    
    def join_queue(self, station):
        """Join the queue at a station"""
        self.assigned_station = station
        self.in_queue = True
        self.queue_arrival_time = datetime.now()
        
        # Log queue join
        self._log_event("Joined Queue", {
            "station_id": station.id,
            "location": f"({station.location[0]:.6f}, {station.location[1]:.6f})",
            "queue_length": station.get_queue_length(),
            "estimated_wait": f"{station.get_current_wait_time_estimate()} seconds"
        })
    
    def update_waiting_time(self, time_step_seconds):
        """Update waiting time for EV in queue"""
        if self.in_queue:
            self.waiting_time += time_step_seconds

    def advance_travel_to_charger(self, time_step_seconds):
        """Advance simulated travel time toward an assigned charging station.

        Returns True once the EV has arrived (travel_time_remaining hit 0)
        and should join `self.pending_station`'s queue; the caller is
        responsible for the actual `add_to_queue` call.
        """
        if not self.en_route_to_charger:
            return False

        self.travel_time_remaining -= time_step_seconds
        if self.travel_time_remaining <= 0:
            self.en_route_to_charger = False
            return True
        return False
    
    def calculate_energy_needed_for_destination(self):
        """Calculate energy needed to reach destination from current position"""
        try:
            # Calculate remaining distance to destination
            remaining_distance = self.calculate_remaining_distance()
            
            # Calculate energy needed
            energy_needed = remaining_distance * self.consumption_rate
            
            # Add buffer (10 km worth of energy)
            buffer_energy = 10 * self.consumption_rate
            
            return energy_needed + buffer_energy
        except Exception as e:
            # Default to a reasonable value if calculation fails
            return 0.5 * self.battery_capacity  # Charge to 50% as fallback
    
    def calculate_remaining_distance(self):
        """Calculate remaining distance to destination"""
        try:
            if not self.route or len(self.route) < 2:
                return 0
            
            if self.route_index >= len(self.route) - 1:
                return 0
            
            # Ensure route_index is valid
            safe_index = min(max(0, self.route_index), len(self.route) - 1)
            
            total_distance = 0
            for i in range(safe_index, len(self.route) - 1):
                dist = self._calculate_distance(self.route[i], self.route[i + 1])
                # Check for invalid distance calculation
                if dist < 0 or dist > 1000:  # Sanity check: no segment should be >1000km
                    dist = 0
                total_distance += dist
            
            return total_distance
        except Exception as e:
            # Return a default distance if calculation fails
            return 50  # Default 50km as fallback
    
    def calculate_total_route_distance(self):
        """Calculate total distance of the route"""
        try:
            if not self.route or len(self.route) < 2:
                return 0
            
            total_distance = 0
            for i in range(len(self.route) - 1):
                dist = self._calculate_distance(self.route[i], self.route[i + 1])
                # Check for invalid distance calculation
                if dist < 0 or dist > 1000:  # Sanity check: no segment should be >1000km
                    dist = 0
                total_distance += dist
            
            return total_distance
        except Exception as e:
            # Return a default distance if calculation fails
            return 50  # Default 50km as fallback
    
    def calculate_energy_for_total_route(self):
        """Calculate energy needed for the total route"""
        total_distance = self.calculate_total_route_distance()
        return total_distance * self.consumption_rate
    
    def calculate_target_soc(self, queue_length):
        """
        Calculate target SoC based on queue length and energy needed
        
        If queue is empty, can charge to 100%
        Otherwise, charge only enough to reach destination + 10km buffer,
        but not more than 80%
        """
        if queue_length == 0:
            return 1.0  # Charge to 100% if queue is empty
        
        # Calculate energy needed to reach destination plus buffer
        energy_needed = self.calculate_energy_needed_for_destination()
        
        # Calculate SoC needed
        soc_needed = min(0.8, self.soc + (energy_needed / self.battery_capacity))
        
        return soc_needed
    
    def charge(self, time_step_seconds, charging_rate, queue_length):
        """Charge the EV for one time step"""
        if not self.charging:
            return

        self.charging_time_elapsed += time_step_seconds

        # Calculate target SoC based on queue length and energy needed
        self.target_soc = self.calculate_target_soc(queue_length)
        
        # Calculate energy received in this time step (kWh)
        energy_received = charging_rate * (time_step_seconds / 3600)  # Convert seconds to hours
        
        # Update SOC
        old_soc = self.soc
        new_soc = self.soc + (energy_received / self.battery_capacity)
        self.soc = min(new_soc, self.target_soc)  # Cap at target SoC
        
        # Check if charging is complete (reached target SoC)
        if self.soc >= self.target_soc:
            self._log_event("Charging Complete", {
                "battery_before": f"{old_soc * 100:.1f}%",
                "battery_after": f"{self.soc * 100:.1f}%",
                "target_battery": f"{self.target_soc * 100:.1f}%",
                "energy_added": f"{(self.soc - old_soc) * self.battery_capacity:.2f} kWh",
                "charging_duration": f"{(datetime.now() - self.charging_start_time).total_seconds()} seconds"
            })
            self.finish_charging()
        elif time_step_seconds > 0:  # Only log if meaningful time has passed
            # Log charging progress
            self._log_event("Charging Progress", {
                "battery": f"{self.soc * 100:.1f}%",
                "target": f"{self.target_soc * 100:.1f}%",
                "energy_added": f"{(self.soc - old_soc) * self.battery_capacity:.2f} kWh"
            })
    
    def finish_charging(self):
        """Finish charging and continue journey"""
        self.charging = False
        self.assigned_station = None
        self.waiting_time = 0  # Reset waiting time when charging completes
        self.completed_charging_duration = self.charging_time_elapsed
        self.charging_time_elapsed = 0.0
    
    def abandon(self, reason):
        """Mark EV as abandoned due to unsolvable situation"""
        self.abandoned = True
        self._log_event("Abandoned", {
            "reason": reason,
            "battery": f"{self.soc * 100:.1f}%",
            "position": f"({self.current_position[0]:.6f}, {self.current_position[1]:.6f})",
            "remaining_distance": f"{self.calculate_remaining_distance():.2f} km"
        })
    
    def can_reach_station(self, station_location):
        """Check if the EV can reach the station with current battery"""
        try:
            distance = self._calculate_distance(self.current_position, station_location)
            # Sanity check on distance
            if distance < 0 or distance > 1000:  # No station should be >1000km away
                distance = 50  # Default reasonable distance
            
            energy_required = distance * self.consumption_rate
            return self.soc * self.battery_capacity >= energy_required
        except Exception as e:
            # Default to saying we can reach it, and let other checks determine suitability
            return True
    
    def is_station_on_route(self, station_location, max_detour=1000):
        """
        Check if station is on or close to the route
        
        Args:
            station_location: (lat, lng) of the station
            max_detour: Maximum allowed detour in meters
            
        Returns:
            bool: True if station is on route within max_detour
        """
        try:
            if not self.route or self.route_index >= len(self.route):
                return False
            
            # Ensure route_index is valid
            safe_index = min(max(0, self.route_index), len(self.route) - 1)
            
            # Check distance from each point in remaining route to station
            for i in range(safe_index, len(self.route)):
                route_point = self.route[i]
                try:
                    distance_to_station = calculate_distance(route_point, station_location)
                    
                    if distance_to_station <= max_detour:
                        return True
                except Exception:
                    # Skip this point if calculation fails
                    continue
            
            return False
        except Exception as e:
            # Default to False if overall calculation fails
            return False
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points in km.

        Uses the same haversine formula as `models.maps_service.calculate_distance`
        (used by optimization/scoring/policies) so reachability checks and
        station scoring always agree on distance.
        """
        try:
            lat1, lng1 = point1
            lat2, lng2 = point2
            
            # Validate inputs
            if not all(isinstance(coord, (int, float)) for coord in [lat1, lng1, lat2, lng2]):
                return 0
            
            distance = calculate_distance(point1, point2) / 1000.0  # meters -> km
            
            # Sanity check the result
            if distance < 0 or distance > 1000:  # No segment should be >1000km
                return 0
            
            return distance
        except Exception as e:
            # Return a small non-zero value as fallback
            return 0.1  # 100m as default
        
    def to_dict(self):
        """Convert EV to dictionary for API response"""
        return {
            'id': self.id,
            'current_position': self.current_position,
            'soc': self.soc,
            'target_soc': self.target_soc,
            'charging': self.charging,
            'in_queue': self.in_queue,
            'en_route_to_charger': self.en_route_to_charger,
            'assigned_station': self.assigned_station.id if self.assigned_station else None,
            'waiting_time': self.waiting_time,
            'trip_completed': self.trip_completed,
            'abandoned': self.abandoned,
            'journey_log': self.journey_log
        }
