import time
import threading
from datetime import datetime
import config
from models.optimization import optimize_charging, get_optimization_logs

class Simulation:
    def __init__(self, evs=None, stations=None, routes=None):
        self.evs = evs or []
        self.stations = stations or []
        self.routes = routes or []
        self.time_step = config.TIME_STEP_SECONDS
        self.current_step = 0
        self.running = False
        self.thread = None
        self.metrics = {
            'average_wait_time': 0,
            'average_detour_distance': 0,
            'max_queue_length': 0,
            'station_utilization': {},
            'completion_rate': 0,
            'abandoned_rate': 0,  # Added metric for abandoned EVs
            'optimization_time': 0
        }
        self.step_history = []
        self.last_optimization_step = -config.OPTIMIZATION_INTERVAL  # Force initial optimization
        self.optimization_logs = []
        
        # Track stalled EVs for monitoring
        self.stalled_positions = {}  # EV ID -> {position, stall_count}
        self.last_optimization_error = None
    
    def start(self):
        """Start the simulation in a separate thread"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        return True
    
    def _run_simulation(self):
        """Main simulation loop"""
        while self.running:
            # Run one step
            self.step()
            
            # Record state for history
            self._record_state()
            
            # Sleep to control simulation speed (real-time factor)
            time.sleep(0.1)  # 10 steps per second regardless of time_step value
    
    def step(self):
        """Run one simulation step"""
        try:
            # Update EVs
            for ev in self.evs:
                if not ev.abandoned:  # Skip abandoned EVs
                    prev_position = ev.current_position
                    prev_index = ev.route_index
                    
                    if not (ev.charging or ev.in_queue):
                        ev.move(self.time_step)
                        
                    # Monitor for stalled EVs (not moving, not charging, not in queue, not completed)
                    if (not ev.trip_completed and 
                        not ev.charging and 
                        not ev.in_queue and 
                        prev_position == ev.current_position and 
                        prev_index == ev.route_index):
                        
                        # Initialize or increment stall counter
                        if ev.id not in self.stalled_positions:
                            self.stalled_positions[ev.id] = {'position': prev_position, 'count': 1}
                        else:
                            self.stalled_positions[ev.id]['count'] += 1
                        
                        # If stalled for too long, reset route index or increase battery
                        if self.stalled_positions[ev.id]['count'] > 10:  # Stalled for 10 steps
                            if ev.soc < 0.1:
                                # If battery is low, boost it to continue journey
                                ev.soc = min(ev.soc + 0.2, 0.5)  # Boost to at least 50% if very low
                                print(f"Boosted battery for stalled EV {ev.id}: {ev.soc}")
                            
                            # Reset stall counter
                            self.stalled_positions[ev.id]['count'] = 0
                    else:
                        # Reset stall counter if moving
                        if ev.id in self.stalled_positions:
                            del self.stalled_positions[ev.id]
            
            # Update stations
            for station in self.stations:
                station.update(self.time_step)
            
            # Find EVs that need charging
            evs_needing_charge = [ev for ev in self.evs if not ev.abandoned and ev.needs_charging(config.CHARGE_THRESHOLD)]
            
            # Run optimization if needed
            if (len(evs_needing_charge) > 0 and
                (self.current_step - self.last_optimization_step >= config.OPTIMIZATION_INTERVAL)):
                
                try:
                    self._run_optimization(evs_needing_charge)
                    self.last_optimization_error = None
                except Exception as e:
                    self.last_optimization_error = str(e)
                    print(f"Optimization error: {e}")
                    
                self.last_optimization_step = self.current_step
            
            # Update metrics
            self._update_metrics()
            
            # Increment step
            self.current_step += 1
        except Exception as e:
            print(f"Error in simulation step: {e}")
    
    def _run_optimization(self, evs_needing_charge):
        """Run optimization algorithm and assign stations"""
        try:
            start_time = time.time()
            
            # Run optimizer - now returns assignments and abandoned EVs
            assignments, abandoned_evs = optimize_charging(evs_needing_charge, self.stations)
            
            # Record optimization time
            optimization_time = time.time() - start_time
            self.metrics['optimization_time'] = optimization_time
            
            # Get optimization logs
            self.optimization_logs = get_optimization_logs()
            
            # Apply assignments
            for ev in evs_needing_charge:
                if ev.id in assignments:
                    station_id = assignments[ev.id]
                    # Find station
                    for station in self.stations:
                        if station.id == station_id:
                            # Add EV to station queue
                            station.add_to_queue(ev)
                            break
                elif ev.id in abandoned_evs:
                    # Mark EV as abandoned
                    ev.abandon("No reachable charging station with current battery")
                    print(f"EV {ev.id} abandoned due to unsolvable charging situation")
        except Exception as e:
            print(f"Optimization error: {e}")
            self.optimization_logs.append(f"Optimization error: {e}")
    
    def _update_metrics(self):
        """Update simulation metrics"""
        # Calculate metrics
        total_wait_time = sum(ev.waiting_time for ev in self.evs if ev.waiting_time > 0)
        evs_with_wait = sum(1 for ev in self.evs if ev.waiting_time > 0)
        
        # Use CURRENT maximum queue length instead of historical maximum
        current_max_queue = max((len(station.queue) for station in self.stations), default=0)
        
        completed_trips = sum(1 for ev in self.evs if ev.trip_completed)
        abandoned_evs = sum(1 for ev in self.evs if ev.abandoned)
        
        completion_rate = completed_trips / len(self.evs) if len(self.evs) > 0 else 0
        abandoned_rate = abandoned_evs / len(self.evs) if len(self.evs) > 0 else 0
        
        # Update metrics dict
        self.metrics['average_wait_time'] = total_wait_time / evs_with_wait if evs_with_wait > 0 else 0
        self.metrics['max_queue_length'] = current_max_queue  # Use current maximum
        self.metrics['completion_rate'] = completion_rate
        self.metrics['abandoned_rate'] = abandoned_rate
        
        # Station utilization
        for station in self.stations:
            self.metrics['station_utilization'][station.id] = len(station.charging_evs) / station.num_chargers
    
    def _build_state(self):
        """Build a serializable snapshot of the current simulation state."""
        return {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'evs': [ev.to_dict() for ev in self.evs],
            'stations': [station.to_dict() for station in self.stations],
            'metrics': self.metrics.copy(),
            'optimization_logs': self.optimization_logs.copy()
        }

    def _record_state(self):
        """Record current state for history"""
        state = self._build_state()
        self.step_history.append(state)
        
        # Keep only recent history to avoid memory issues
        if len(self.step_history) > 1000:
            self.step_history = self.step_history[-1000:]
    
    def get_current_state(self):
        """Get current simulation state"""
        if not self.step_history:
            return self._build_state()
        return self.step_history[-1]
    
    def get_history(self, start=0, count=100):
        """Get simulation history"""
        if start >= len(self.step_history):
            return []
        
        end = min(start + count, len(self.step_history))
        return self.step_history[start:end]
    
    def get_optimization_logs(self):
        """Get current optimization logs"""
        return self.optimization_logs
    
    def get_ev_journey_log(self, ev_id):
        """Get journey log for a specific EV"""
        for ev in self.evs:
            if ev.id == ev_id:
                return ev.journey_log
        return []
    
    def reset(self):
        """Reset simulation to initial state"""
        self.stop()
        
        # Reset EVs
        for ev in self.evs:
            ev.current_position = ev.origin
            ev.route_index = 0
            ev.soc = getattr(ev, 'initial_soc', ev.soc)
            ev.charging = False
            ev.in_queue = False
            ev.assigned_station = None
            ev.queue_arrival_time = None
            ev.charging_start_time = None
            ev.waiting_time = 0
            ev.target_soc = 0.8
            ev.trip_completed = False
            ev.abandoned = False
            ev.trip_start_time = datetime.now()
            ev.trip_end_time = None
            ev.journey_log = []
            # Record initialization
            ev._log_event("Initialized", {
                "origin_node": f"Node at {ev.origin}",
                "destination_node": f"Node at {ev.destination}",
                "battery": f"{ev.soc * 100:.1f}%",
                "total_distance": ev.calculate_total_route_distance(),
                "battery_required": f"{ev.calculate_energy_for_total_route() / ev.battery_capacity * 100:.1f}%"
            })
        
        # Reset stations
        for station in self.stations:
            station.charging_evs = []
            station.queue.clear()
            station.total_served = 0
            station.total_wait_time = 0
            station.max_queue_length = 0
        
        self.current_step = 0
        self.step_history = []
        self.optimization_logs = []
        self.last_optimization_step = -config.OPTIMIZATION_INTERVAL
        
        # Reset stalled EVs
        self.stalled_positions = {}
        self.last_optimization_error = None
        
        return True
