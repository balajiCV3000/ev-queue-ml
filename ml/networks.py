"""Neural network architectures for RL training."""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ml.features import NUM_FEATURES


def build_mlp(input_dim=NUM_FEATURES, hidden=64):
    """Build MLP NUM_FEATURES->64->64->1 Q-network."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for training; install from requirements-ml.txt")
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def numpy_forward(weights, features):
    """
    Pure-numpy forward pass for serving.

    weights dict keys: w1, b1, w2, b2, w3, b3
    """
    import numpy as np
    x = features
    x = np.maximum(0, x @ weights["w1"] + weights["b1"])
    x = np.maximum(0, x @ weights["w2"] + weights["b2"])
    return (x @ weights["w3"] + weights["b3"]).flatten()


def export_weights(model):
    """Export PyTorch model weights to numpy dict for .npz serving."""
    import numpy as np
    layers = [m for m in model if hasattr(m, 'weight')]
    return {
        "w1": layers[0].weight.detach().cpu().numpy().T,
        "b1": layers[0].bias.detach().cpu().numpy(),
        "w2": layers[1].weight.detach().cpu().numpy().T,
        "b2": layers[1].bias.detach().cpu().numpy(),
        "w3": layers[2].weight.detach().cpu().numpy().T,
        "b3": layers[2].bias.detach().cpu().numpy(),
    }


def save_npz(path, weights):
    """Save weights to .npz file."""
    import numpy as np
    np.savez(path, **weights)


def load_npz(path):
    """Load weights from .npz file."""
    import numpy as np
    data = np.load(path)
    return {k: data[k] for k in data.files}


def init_numpy_weights(input_dim=NUM_FEATURES, hidden=64, seed=42):
    """Initialize random weights for smoke model without torch."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / input_dim)
    scale2 = np.sqrt(2.0 / hidden)
    return {
        "w1": (rng.standard_normal((input_dim, hidden)) * scale1).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.standard_normal((hidden, hidden)) * scale2).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "w3": (rng.standard_normal((hidden, 1)) * scale2).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }
