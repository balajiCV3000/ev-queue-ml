import uuid
from collections import deque

class ChargingStation:
    def __init__(self, id=None, location=None, num_chargers=2, charging_rate=7.0):
        self.id = id or str(uuid.uuid4())
        self.location = location  # (lat, lng)
        self.num_chargers = num_chargers
        self.charging_rate = charging_rate  # kW
        
        self.charging_evs = []  # EVs currently charging
        self.queue = deque()  # EVs waiting to charge
        
        self.total_served = 0  # Total number of EVs served
        self.total_wait_time = 0  # Total wait time of all EVs
        self.max_queue_length = 0  # Maximum queue length observed
    
    def add_to_queue(self, ev):
        """Add an EV to the charging queue"""
        self.queue.append(ev)
        ev.join_queue(self)
        
        if len(self.queue) > self.max_queue_length:
            self.max_queue_length = len(self.queue)
    
    def start_next_in_queue(self):
        """Start charging the next EV in queue if charger available"""
        if not self.queue or len(self.charging_evs) >= self.num_chargers:
            return False
        
        # Get next EV from queue
        ev = self.queue.popleft()
        
        # Update wait time statistics
        wait_time = ev.waiting_time
        self.total_wait_time += wait_time
        
        # Start charging
        ev.start_charging(self)
        self.charging_evs.append(ev)
        self.total_served += 1
        
        return True
    
    def update(self, time_step_seconds):
        """Update station for one time step"""
        # Update EVs currently charging
        evs_finished = []
        queue_length = len(self.queue)
        
        for ev in self.charging_evs:
            # Pass queue length to determine charging limit
            ev.charge(time_step_seconds, self.charging_rate, queue_length)
            if not ev.charging:
                evs_finished.append(ev)
        
        for ev in evs_finished:
            self.charging_evs.remove(ev)
            
        # Start charging EVs from queue if possible
        while len(self.charging_evs) < self.num_chargers and self.queue:
            self.start_next_in_queue()
        
        for ev in self.queue:
            ev.update_waiting_time(time_step_seconds)
    
    def get_current_wait_time_estimate(self):
        """Estimate wait time for a new arrival (in seconds)"""
        if len(self.charging_evs) < self.num_chargers:
            return 0  # No wait if charger available
        
        # Estimate charging time based on current EVs and queue
        total_waiting_time = 0
        
        # Check each occupied charger
        for ev in self.charging_evs:
            # Estimate remaining charge time
            energy_to_charge = (ev.target_soc - ev.soc) * ev.battery_capacity
            time_remaining = (energy_to_charge / self.charging_rate) * 3600  # Convert hours to seconds
            total_waiting_time += time_remaining
        
        # Add estimated time for queued EVs
        for ev in self.queue:
            # Rough estimate based on charging from current SoC to 80%
            energy_to_charge = (0.8 - ev.soc) * ev.battery_capacity
            charge_time = (energy_to_charge / self.charging_rate) * 3600  # Convert hours to seconds
            total_waiting_time += charge_time
        
        # Average across chargers
        if self.num_chargers > 0:
            average_wait = total_waiting_time / self.num_chargers
            
            # Calculate queue position
            queue_position = len(self.queue) + 1  # +1 for the new arrival
            
            # Calculate how many charging cycles needed
            cycles_needed = (queue_position + self.num_chargers - 1) // self.num_chargers
            
            return average_wait * cycles_needed
        
        return 0
    
    def get_queue_length(self):
        """Get current queue length"""
        return len(self.queue)
    
    def to_dict(self):
        """Convert station to dictionary for API response"""
        return {
            'id': self.id,
            'location': self.location,
            'num_chargers': self.num_chargers,
            'charging_rate': self.charging_rate,
            'charging_evs': [ev.id for ev in self.charging_evs],
            'queue_length': len(self.queue),  # Current queue length
            'total_served': self.total_served,
            'average_wait_time': self.total_wait_time / self.total_served if self.total_served > 0 else 0
        }