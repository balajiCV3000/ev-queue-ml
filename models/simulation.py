import time
import threading
from datetime import datetime

import config
from models.maps_service import calculate_distance
from models.optimization import get_optimization_logs
from ml.policies.registry import get_policy
from ml.tracking import RewardTracker


class Simulation:
    def __init__(self, evs=None, stations=None, routes=None, policy=None):
        self.evs = evs or []
        self.stations = stations or []
        self.routes = routes or []
        self.time_step = config.TIME_STEP_SECONDS
        self.current_step = 0
        self.running = False
        self.completed = False
        self.thread = None
        self.metrics = {
            'average_wait_time': 0,
            'max_queue_length': 0,
            'avg_queue_length': 0,
            'station_utilization': {},
            'station_utilization_time_avg': {},
            'avg_station_utilization': 0,
            'completion_rate': 0,
            'abandoned_rate': 0,
            'optimization_time': 0,
            'total_travel_distance_km': 0,
            'average_travel_distance_km': 0,
            'average_charging_completion_time': 0,
            'vehicles_served': 0,
            'throughput_per_hour': 0,
        }
        self.step_history = []
        self.last_optimization_step = -config.OPTIMIZATION_INTERVAL
        self.optimization_logs = []

        self.stalled_positions = {}
        self.last_optimization_error = None

        # Episode-long running trackers used to compute metrics that must be
        # accumulated over time rather than read as a final-step snapshot.
        self._peak_queue_length = 0
        self._queue_length_sum = 0.0
        self._queue_samples = 0
        self._total_travel_distance_km = 0.0
        self._num_travel_assignments = 0
        self._station_utilization_sum = {}
        self._utilization_samples = 0
        self._charging_completion_times = []
        self._counted_charging_ev_ids = set()

        self.policy = policy or get_policy(config.ASSIGNMENT_POLICY)
        self.policy_name = getattr(self.policy, 'name', config.ASSIGNMENT_POLICY)
        self.reward_tracker = RewardTracker()

    def start(self):
        """Start the simulation in a separate thread"""
        if self.running:
            return False
        if self.completed:
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
        import time as _time
        while self.running and not self.is_done():
            self.step()
            self._record_state()
            _time.sleep(0.1)

        if self.running:
            # Loop exited because is_done() became true, not because stop()
            # was called — surface completion and capture the final snapshot.
            self.completed = True
            self.running = False
            self._record_state()

    def get_evs_needing_charge(self):
        """Return EVs that need charging and are not abandoned."""
        return [
            ev for ev in self.evs
            if not ev.abandoned and ev.needs_charging(config.CHARGE_THRESHOLD)
        ]

    def optimization_due(self):
        """Check whether station assignment should run this step."""
        return (
            self.current_step - self.last_optimization_step >= config.OPTIMIZATION_INTERVAL
        )

    def is_done(self):
        """True when all EVs have completed or been abandoned."""
        if not self.evs:
            return True
        return all(ev.trip_completed or ev.abandoned for ev in self.evs)

    def step(self, auto_optimize=True):
        """Run one simulation step. Returns step reward from RewardTracker."""
        try:
            for ev in self.evs:
                if not ev.abandoned:
                    prev_position = ev.current_position
                    prev_index = ev.route_index

                    if ev.en_route_to_charger:
                        arrived = ev.advance_travel_to_charger(self.time_step)
                        if arrived and ev.pending_station is not None:
                            station = ev.pending_station
                            ev.pending_station = None
                            station.add_to_queue(ev)
                    elif not (ev.charging or ev.in_queue):
                        ev.move(self.time_step)

                    if config.ENABLE_STALL_RESCUE:
                        if (not ev.trip_completed and
                            not ev.charging and
                            not ev.in_queue and
                            not ev.en_route_to_charger and
                            prev_position == ev.current_position and
                            prev_index == ev.route_index):

                            if ev.id not in self.stalled_positions:
                                self.stalled_positions[ev.id] = {'position': prev_position, 'count': 1}
                            else:
                                self.stalled_positions[ev.id]['count'] += 1

                            if self.stalled_positions[ev.id]['count'] > 10:
                                if ev.soc < 0.1:
                                    ev.soc = min(ev.soc + 0.2, 0.5)
                                    print(f"Boosted battery for stalled EV {ev.id}: {ev.soc}")
                                self.stalled_positions[ev.id]['count'] = 0
                        else:
                            if ev.id in self.stalled_positions:
                                del self.stalled_positions[ev.id]

            for station in self.stations:
                station.update(self.time_step)

            evs_needing_charge = self.get_evs_needing_charge()

            if auto_optimize and evs_needing_charge and self.optimization_due():
                try:
                    self._run_optimization(evs_needing_charge)
                    self.last_optimization_error = None
                except Exception as e:
                    self.last_optimization_error = str(e)
                    print(f"Optimization error: {e}")
                self.last_optimization_step = self.current_step

            self._update_metrics()
            step_reward = self.reward_tracker.observe(
                self.evs, self.stations, self.time_step
            )
            self.current_step += 1
            return step_reward
        except Exception as e:
            print(f"Error in simulation step: {e}")
            return 0.0

    def run_headless(self, max_steps=1500):
        """Run simulation without threading, sleep, or state recording."""
        steps_run = 0
        while steps_run < max_steps and not self.is_done():
            self.step(auto_optimize=True)
            steps_run += 1
        return {
            "steps": steps_run,
            "metrics": self.metrics.copy(),
            "reward_summary": self.reward_tracker.summary(),
            "done": self.is_done(),
        }

    def set_policy(self, policy):
        """Hot-swap assignment policy (name string or policy instance)."""
        if isinstance(policy, str):
            self.policy = get_policy(policy)
        else:
            self.policy = policy
        self.policy_name = getattr(self.policy, 'name', 'unknown')

    def get_policy_name(self):
        return self.policy_name

    @property
    def fallback_active(self):
        return getattr(self.policy, 'fallback_active', False)

    def _run_optimization(self, evs_needing_charge):
        """Run policy assignment and apply results."""
        try:
            start_time = time.time()

            ctx = {
                'current_step': self.current_step,
                'time_step': self.time_step,
                'routes': self.routes,
            }
            assignments, abandoned_evs = self.policy.assign(
                evs_needing_charge, self.stations, ctx
            )

            optimization_time = time.time() - start_time
            self.metrics['optimization_time'] = optimization_time
            self.optimization_logs = get_optimization_logs()

            for ev in evs_needing_charge:
                if ev.id in assignments:
                    station_id = assignments[ev.id]
                    for station in self.stations:
                        if station.id == station_id:
                            self.assign_ev_to_station(ev, station)
                            break
                elif ev.id in abandoned_evs:
                    ev.abandon("No reachable charging station with current battery")
                    print(f"EV {ev.id} abandoned due to unsolvable charging situation")
        except Exception as e:
            print(f"Optimization error: {e}")
            self.optimization_logs.append(f"Optimization error: {e}")

    def assign_ev_to_station(self, ev, station):
        """Apply a single EV -> station assignment.

        Shared by `_run_optimization` (auto_optimize=True path) and
        `ml.env.EnvWrapper.step()` (explicit-assignments/RL/eval path) so
        travel-distance accounting, travel energy, and the flag-gated
        travel-then-queue delay are applied consistently regardless of which
        caller drives the simulation.
        """
        distance_km = calculate_distance(ev.current_position, station.location) / 1000.0
        ev.travel_to_charger_km += distance_km
        self._total_travel_distance_km += distance_km
        self._num_travel_assignments += 1

        if config.SIMULATE_TRAVEL_ENERGY:
            self._deduct_travel_energy(ev, station)

        if config.SIMULATE_TRAVEL_TIME:
            travel_seconds = (distance_km / config.TRAVEL_SPEED_KMH) * 3600.0
            ev.en_route_to_charger = True
            ev.pending_station = station
            ev.travel_time_remaining = travel_seconds
            ev._log_event("Traveling to Station", {
                "station_id": station.id,
                "distance_km": f"{distance_km:.2f}",
                "eta_seconds": f"{travel_seconds:.0f}",
            })
        else:
            station.add_to_queue(ev)

    def _deduct_travel_energy(self, ev, station):
        """Deduct battery energy for travel to assigned station."""
        travel_km = calculate_distance(ev.current_position, station.location) / 1000.0
        energy = travel_km * ev.consumption_rate
        ev.soc = max(0.0, ev.soc - energy / ev.battery_capacity)

    def _update_metrics(self):
        """Update simulation metrics.

        Notes on definitions (see plan doc for rationale):
        - average_wait_time: fleet-wide mean wait time over ALL EVs that have
          actually been served (started charging), i.e.
          sum(station.total_wait_time) / sum(station.total_served) across all
          stations. This is captured at the moment each EV leaves the queue
          (before EV.finish_charging() resets its waiting_time to 0), so it
          reflects every served EV rather than only the (near-empty) set of
          EVs with a positive `waiting_time` snapshot at this instant.
        - max_queue_length / avg_queue_length: the fleet-wide queued-vehicle
          count (sum of all station queue lengths) is sampled every step and
          used to track a running peak and a running time-average across the
          whole episode, instead of a single final-step snapshot.
        - total/average_travel_distance_km: accumulated at assignment time in
          `_run_optimization` (distance from the EV's position to its
          assigned station), regardless of whether SIMULATE_TRAVEL_TIME is on.
        - average_charging_completion_time: mean simulated charging duration
          (EV.charging_time_elapsed at the moment charging finishes) over all
          EVs that have finished at least one charging cycle.
        - station_utilization_time_avg / avg_station_utilization: per-station
          and fleet-wide utilization (charging_evs / num_chargers) averaged
          over every step of the episode, complementing the existing
          instantaneous `station_utilization` snapshot (kept for the live UI).
        - vehicles_served / throughput_per_hour: count of EVs that have
          started charging (station.total_served) and that count normalized
          to a per-simulated-hour rate.
        """
        completed_trips = sum(1 for ev in self.evs if ev.trip_completed)
        abandoned_evs = sum(1 for ev in self.evs if ev.abandoned)

        completion_rate = completed_trips / len(self.evs) if len(self.evs) > 0 else 0
        abandoned_rate = abandoned_evs / len(self.evs) if len(self.evs) > 0 else 0

        total_served = sum(station.total_served for station in self.stations)
        total_wait_time_served = sum(station.total_wait_time for station in self.stations)
        average_wait_time = total_wait_time_served / total_served if total_served > 0 else 0

        current_total_queue = sum(len(station.queue) for station in self.stations)
        self._peak_queue_length = max(self._peak_queue_length, current_total_queue)
        self._queue_length_sum += current_total_queue
        self._queue_samples += 1
        avg_queue_length = (
            self._queue_length_sum / self._queue_samples if self._queue_samples > 0 else 0
        )

        average_travel_distance_km = (
            self._total_travel_distance_km / self._num_travel_assignments
            if self._num_travel_assignments > 0 else 0
        )

        for ev in self.evs:
            if (ev.completed_charging_duration is not None
                    and ev.id not in self._counted_charging_ev_ids):
                self._charging_completion_times.append(ev.completed_charging_duration)
                self._counted_charging_ev_ids.add(ev.id)
        average_charging_completion_time = (
            sum(self._charging_completion_times) / len(self._charging_completion_times)
            if self._charging_completion_times else 0
        )

        self._utilization_samples += 1
        for station in self.stations:
            instant_util = (
                len(station.charging_evs) / station.num_chargers if station.num_chargers > 0 else 0
            )
            self.metrics['station_utilization'][station.id] = instant_util
            self._station_utilization_sum[station.id] = (
                self._station_utilization_sum.get(station.id, 0.0) + instant_util
            )

        utilization_time_avg = {
            station_id: total / self._utilization_samples
            for station_id, total in self._station_utilization_sum.items()
        }
        avg_station_utilization = (
            sum(utilization_time_avg.values()) / len(utilization_time_avg)
            if utilization_time_avg else 0
        )

        elapsed_hours = ((self.current_step + 1) * self.time_step) / 3600.0
        throughput_per_hour = total_served / elapsed_hours if elapsed_hours > 0 else 0

        self.metrics['average_wait_time'] = average_wait_time
        self.metrics['max_queue_length'] = self._peak_queue_length
        self.metrics['avg_queue_length'] = avg_queue_length
        self.metrics['completion_rate'] = completion_rate
        self.metrics['abandoned_rate'] = abandoned_rate
        self.metrics['total_travel_distance_km'] = self._total_travel_distance_km
        self.metrics['average_travel_distance_km'] = average_travel_distance_km
        self.metrics['average_charging_completion_time'] = average_charging_completion_time
        self.metrics['station_utilization_time_avg'] = utilization_time_avg
        self.metrics['avg_station_utilization'] = avg_station_utilization
        self.metrics['vehicles_served'] = total_served
        self.metrics['throughput_per_hour'] = throughput_per_hour

    def _build_state(self):
        """Build a serializable snapshot of the current simulation state."""
        if self.completed:
            status = 'completed'
        elif self.running:
            status = 'running'
        else:
            status = 'stopped'
        return {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'status': status,
            'evs': [ev.to_dict() for ev in self.evs],
            'stations': [station.to_dict() for station in self.stations],
            'metrics': self.metrics.copy(),
            'optimization_logs': self.optimization_logs.copy(),
            'policy': self.policy_name,
            'fallback_active': self.fallback_active,
        }

    def _record_state(self):
        """Record current state for history"""
        state = self._build_state()
        self.step_history.append(state)

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
        self.completed = False

        for ev in self.evs:
            ev.current_position = ev.origin
            ev.route_index = 0
            ev.soc = getattr(ev, 'initial_soc', ev.soc)
            ev.charging = False
            ev.in_queue = False
            ev.en_route_to_charger = False
            ev.pending_station = None
            ev.travel_time_remaining = 0.0
            ev.travel_to_charger_km = 0.0
            ev.charging_time_elapsed = 0.0
            ev.completed_charging_duration = None
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
            ev._log_event("Initialized", {
                "origin_node": f"Node at {ev.origin}",
                "destination_node": f"Node at {ev.destination}",
                "battery": f"{ev.soc * 100:.1f}%",
                "total_distance": ev.calculate_total_route_distance(),
                "battery_required": f"{ev.calculate_energy_for_total_route() / ev.battery_capacity * 100:.1f}%"
            })

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
        self.stalled_positions = {}
        self.last_optimization_error = None
        self.reward_tracker.reset()

        self._peak_queue_length = 0
        self._queue_length_sum = 0.0
        self._queue_samples = 0
        self._total_travel_distance_km = 0.0
        self._num_travel_assignments = 0
        self._station_utilization_sum = {}
        self._utilization_samples = 0
        self._charging_completion_times = []
        self._counted_charging_ev_ids = set()

        return True
