SCHEDULER_REGISTRY = {
    "StepLR": {"step_size": 30, "gamma": 0.1},
    "MultiStepLR": {"milestones": [30, 80], "gamma": 0.1},
    "ExponentialLR": {"gamma": 0.95},
    "CosineAnnealingLR": {"T_max": 50, "eta_min": 0},
    "ReduceLROnPlateau": {"mode": "min", "factor": 0.1, "patience": 10},
}
