"""Observational reward tracking without mutating EV/station models.

Note: travel distance, peak/avg queue length, time-averaged station
utilization, charging completion time, and throughput are tracked in
`models.simulation.Simulation._update_metrics`/`Simulation.metrics` rather
than here, since they depend on station/EV state that already lives on the
Simulation and its stations (see docstring there). This tracker stays
focused on the reward-relevant realized costs (travel/wait/charge/system
time, completion/abandonment) that are used to compute a scalar RL reward
signal. `ml.evaluate.run_single` merges both sources into one
comparison-table row.

`total_system` (and therefore the reward returned by `observe`) must include
every second of a trip that a station assignment can make longer, or an RL
policy trained against it has no incentive to avoid picking a station
needlessly far away. It previously summed only queue-wait time + charging
time + a flat abandonment penalty, omitting the `en_route_to_charger` travel
delay entirely (that delay is simulated in
`Simulation.assign_ev_to_station`/`EV.advance_travel_to_charger` whenever
`config.SIMULATE_TRAVEL_TIME` is on, which it is during both RL training and
evaluation). Worse than being merely free, travel *displaced* penalized
time: an EV en route accrues no `waiting_time`, so detouring to a far empty
station strictly reduced the old reward's wait term. That was the root cause
of the RL policy systematically preferring far stations to shave queue time
(avg travel distance ~9-9.7km vs `greedy`'s ~3.7-4.2km, the documented
trade-off in `artifacts/model_card.md` prior to this fix).

`travel_weight` controls how many seconds of system-time cost each second of
en-route travel contributes. At the default 1.0, `total_system` is the true
realized system time (travel + wait + charge + abandonment) -- this is what
evaluation uses, identically for every policy. Training (`ml.train`) passes
a weight > 1 so a detour is only worth taking when it saves clearly more
wait time than the travel it adds, rather than on any positive margin.
"""


class RewardTracker:
    """Per-EV state machine capturing realized travel/wait/charge/system time."""

    def __init__(self, travel_weight=1.0):
        self.travel_weight = travel_weight
        self.reset()

    def reset(self):
        self._prev = {}
        self.total_travel = 0.0
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

            if getattr(ev, "en_route_to_charger", False):
                self.total_travel += time_step
                self.total_system += self.travel_weight * time_step

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
            "total_travel": self.total_travel,
            "total_wait": self.total_wait,
            "total_charge": self.total_charge,
            "total_system": self.total_system,
            "completed": self.completed,
            "abandoned": self.abandoned,
            "reward": -self.total_system,
        }
