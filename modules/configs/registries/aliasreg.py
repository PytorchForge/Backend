UI_ALIASES = {
    "optimizers": {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
        "nadam": "NAdam",
        "radam": "RAdam",
        "lion": "Lion",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad"
    },
    "losses": {
        "crossentropy": "CrossEntropyLoss",
        "focal": "FocalLoss",
        "dice": "DiceLoss",
        "mse": "MSELoss",
        "l1": "L1Loss",
        "bce": "BCELoss",
        "bcewithlogits": "BCEWithLogitsLoss"
    },
    "metrics": {
        "accuracy": "Accuracy",
        "f1": "F1Score",
        "recall": "Recall",
        "mae": "MeanAbsoluteError",
        "mse": "MeanSquaredError",
        "iou": "JaccardIndex"
    },
    "models": {
        "visiontransformer": "ViT",
        "resnet": "ResNet50",
        "resnet18": "ResNet18",
        "efficientnetb0": "EfficientNetB0",
        "efficientnetb7": "EfficientNetB7",
        "fasterrcnn": "FasterRCNN",
        "maskrcnn": "MaskRCNN",
        "deeplabv3": "DeepLabV3",
        "conformer": "Conformer",
        "wave2letter": "Wave2Letter",
        "wavernn": "WaveRNN",
        "glove": "GloVe",
        "fasttext": "FastText",
        "transformerencoderdecoder": "TransformerEncoderDecoder",
        "xlmroberta": "XLMRoberta"
    },
    "layers": {
        "conv1d": "Conv1d",
        "conv2d": "Conv2d",
        "conv3d": "Conv3d",
        "convtranspose1d": "ConvTranspose1d",
        "convtranspose2d": "ConvTranspose2d",
        "convtranspose3d": "ConvTranspose3d",
        "batchnorm1d": "BatchNorm1d",
        "batchnorm2d": "BatchNorm2d",
        "batchnorm3d": "BatchNorm3d",
        "maxpool2d": "MaxPool2d",
        "maxpool1d": "MaxPool1d",
        "maxpool3d": "MaxPool3d",
        "avgpool1d": "AvgPool1d",
        "avgpool2d": "AvgPool2d",
        "avgpool3d": "AvgPool3d",
        "layernorm": "LayerNorm",
        "lstm": "LSTM",
        "dropout": "Dropout",
        "dropout1d": "Dropout1d",
        "dropout2d": "Dropout2d",
        "dropout3d": "Dropout3d",
        "embedding": "Embedding",
        "pixelshuffle": "PixelShuffle",
        "upsample": "Upsample",
        "flatten": "Flatten",
        "unfold": "Unfold",
        "linear": "Linear",
        "bilinear": "Bilinear",
        "transformer": "Transformer",
        "multiheadattention": "MultiheadAttention"
    },
    "subtasks": {
        # Image
        "classification": "classification",
        "object-detection": "object_detection",
        "image-segmentation": "segmentation",
        "generation": "generation",

        # Text
        "text-classification": "classification",
        "summarization": "summarisation",
        "translation": "translation",
        "text-generation": "generation",

        # Audio
        "recognition": "recognition",
        "audio-classification": "classification",
        "audio-generation": "generation",
        "conversion": "conversion"
    },
    "data_formats": {
        "wav": "audio",
        "mp3": "audio",
        "flac": "audio",
        "png": "imagefolder",
        "jpeg": "imagefolder",
        "jpg": "imagefolder",
        "plain-text": "csv",
        "csv": "csv",
        "pickle": "other",
        "pytorch-tensor": "other"
    },
}