"""Policy registry and factory."""

from ml.policies.greedy import GreedyPolicy
from ml.policies.greedy_sequential import GreedySequentialPolicy
from ml.policies.nearest import NearestPolicy
from ml.policies.random import RandomPolicy

_POLICIES = {
    "greedy": GreedyPolicy,
    "greedy_sequential": GreedySequentialPolicy,
    "nearest": NearestPolicy,
    "random": RandomPolicy,
    "rl": None,  # lazy import — needs model path kwargs
}


def list_policies():
    """Return names of registered policies."""
    return sorted(_POLICIES.keys())


def get_policy(name, **kwargs):
    """
    Instantiate a policy by name.

    Falls back to greedy for unknown names.
    """
    cls = _POLICIES.get(name)
    if cls is None:
        if name == "rl":
            from ml.policies.rl_policy import RLPolicy
            return RLPolicy(**kwargs)
        return GreedyPolicy()
    return cls(**kwargs)
