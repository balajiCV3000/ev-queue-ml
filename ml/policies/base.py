"""Assignment policy interface."""


class AssignmentPolicy:
    """Strategy interface for EV-to-station assignment."""

    name = "base"

    def assign(self, evs, stations, ctx):
        """
        Assign EVs to stations without mutating EV or station state.

        Args:
            evs: List of EVs needing charging.
            stations: List of available charging stations.
            ctx: Context dict (current_step, time_step, etc.).

        Returns:
            tuple: (assignments dict {ev_id: station_id}, abandoned_ids list)
        """
        raise NotImplementedError
