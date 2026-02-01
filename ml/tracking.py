"""Observational reward tracking without mutating EV/station models.

Note: travel distance, peak/avg queue length, time-averaged station
utilization, charging completion time, and throughput are tracked in
`models.simulation.Simulation._update_metrics`/`Simulation.metrics` rather
than here, since they depend on station/EV state that already lives on the
Simulation and its stations (see docstring there). This tracker stays
focused on the reward-relevant realized costs (wait/charge/system time,
completion/abandonment) that are used to compute a scalar RL reward signal.
`ml.evaluate.run_single` merges both sources into one comparison-table row.
"""


class RewardTracker:
    """Per-EV state machine capturing realized wait/charge/system time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev = {}
        self.total_wait = 0.0
        self.total_charge = 0.0
        self.total_system = 0.0
        self.completed = 0
        self.abandoned = 0

    def observe(self, evs, stations, time_step):
        """Observe one simulation step and accumulate realized costs."""
        for ev in evs:
            prev = self._prev.get(ev.id, {})
            waiting = ev.waiting_time
            charging = ev.charging

            if prev.get("waiting_time", 0) < waiting:
                delta_wait = waiting - prev.get("waiting_time", 0)
                self.total_wait += delta_wait
                self.total_system += delta_wait

            if charging and not prev.get("charging", False):
                pass

            if charging:
                self.total_charge += time_step
                self.total_system += time_step

            if ev.trip_completed and not prev.get("trip_completed", False):
                self.completed += 1

            if ev.abandoned and not prev.get("abandoned", False):
                self.abandoned += 1
                self.total_system += 3600.0

            self._prev[ev.id] = {
                "waiting_time": waiting,
                "charging": charging,
                "trip_completed": ev.trip_completed,
                "abandoned": ev.abandoned,
            }

        return -self.total_system

    def summary(self):
        """Return aggregate metrics and negative reward (minimize system time)."""
        return {
            "total_wait": self.total_wait,
            "total_charge": self.total_charge,
            "total_system": self.total_system,
            "completed": self.completed,
            "abandoned": self.abandoned,
            "reward": -self.total_system,
        }
