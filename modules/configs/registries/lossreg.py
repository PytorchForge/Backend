LOSS_REGISTRY = {
    "MSELoss": {"reduction": "mean"},
    "L1Loss": {"reduction": "mean"},
    "SmoothL1Loss": {"reduction": "mean", "beta": 1.0},
    "HuberLoss": {"reduction": "mean", "delta": 1.0},
    "CrossEntropyLoss": {"reduction": "mean"},
    "BCELoss": {"reduction": "mean"},
    "BCEWithLogitsLoss": {"reduction": "mean"},
    "NLLLoss": {"reduction": "mean"},
    "MarginRankingLoss": {"reduction": "mean", "margin": 0.0},
    "TripletMarginLoss": {"margin": 1.0, "p": 2, "eps": 1e-6, "swap": False, "reduction": "mean"},
    "CosineEmbeddingLoss": {"margin": 0.0, "reduction": "mean"},
    "MultiMarginLoss": {"p": 1, "margin": 1.0, "reduction": "mean"},
}
