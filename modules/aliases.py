OPTIMIZER_ALIASES = {
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD",
    "nadam": "NAdam",
    "radam": "RAdam",
    "lion": "Lion",  
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
}

LOSS_ALIASES = {
    "crossentropy": "CrossEntropyLoss",
    "focal": "FocalLoss",
    "dice": "DiceLoss",
    "mse": "MSELoss",
    "l1": "L1Loss",
    "bce": "BCELoss",
    "bcewithlogits": "BCEWithLogitsLoss"
}

METRIC_ALIASES = {
    "accuracy": "Accuracy",
    "f1": "F1Score",
    "recall": "Recall",
    "mae": "MeanAbsoluteError",
    "mse": "MeanSquaredError",
    "iou": "JaccardIndex",
}