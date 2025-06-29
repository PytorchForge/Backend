OPTIMIZER_REGISTRY = {
    "SGD": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
    "Adam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False},
    "AdamW": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
    "RMSprop": {"lr": 0.01, "alpha": 0.99, "eps": 1e-8, "momentum": 0.0, "weight_decay": 0.0},
    "Adagrad": {"lr": 0.01, "lr_decay": 0.0, "weight_decay": 0.0, "eps": 1e-10},
    "Adadelta": {"lr": 1.0, "rho": 0.9, "eps": 1e-6, "weight_decay": 0.0},
    "LBFGS": {"lr": 1.0, "max_iter": 20, "max_eval": None, "tolerance_grad": 1e-7, "tolerance_change": 1e-9},
    "NAdam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "momentum_decay": 0.004},
    "RAdam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
    "ASGD": {"lr": 0.01, "lambd": 0.0001, "alpha": 0.75, "t0": 1e6, "weight_decay": 0.0},
}
