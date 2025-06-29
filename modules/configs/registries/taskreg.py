DEFAULTS = {
    "Image Classification": {
        "data_format": "imagefolder",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resize"},
            {"name": "ToTensor"},
            {"name": "Normalize"}
        ]
    },
    "Object Detection": {
        "data_format": "json",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resize"},
            {"name": "ToTensor"}
        ]
    },
    "Image Segmentation": {
        "data_format": "folder",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resize"},
            {"name": "ToTensor"}
        ]
    },
    "Image Generation": {
        "data_format": "imagefolder",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resize"},
            {"name": "ToTensor"}
        ]
    },

    "Text Classification": {
        "data_format": "csv",
        "data_type": "file",
        "preprocessing": [
            {"name": "RegexTokenizer"},
            {"name": "VocabTransform"},
            {"name": "ToTensor"}
        ]
    },
    "Sentiment Analysis": {
        "data_format": "csv",
        "data_type": "file",
        "preprocessing": [
            {"name": "RegexTokenizer"},
            {"name": "VocabTransform"},
            {"name": "ToTensor"}
        ]
    },
    "Named Entity Recognition": {
        "data_format": "json",
        "data_type": "file",
        "preprocessing": [
            {"name": "RegexTokenizer"},
            {"name": "VocabTransform"},
            {"name": "ToTensor"}
        ]
    },
    "Text Generation": {
        "data_format": "json",
        "data_type": "file",
        "preprocessing": [
            {"name": "BARTTokenizer"}
        ]
    },
    "Machine Translation": {
        "data_format": "csv",
        "data_type": "file",
        "preprocessing": [
            {"name": "RegexTokenizer"},
            {"name": "VocabTransform"},
            {"name": "ToTensor"}
        ]
    },
    "Text Summarization": {
        "data_format": "json",
        "data_type": "file",
        "preprocessing": [
            {"name": "BARTTokenizer"}
        ]
    },

    "Speech Recognition": {
        "data_format": "json",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resample"},
            {"name": "MelSpectrogram"},
            {"name": "ToTensor"}
        ]
    },
    "Audio Classification": {
        "data_format": "csv",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resample"},
            {"name": "MFCC"},
            {"name": "ToTensor"}
        ]
    },
    "Audio Generation": {
        "data_format": "other",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resample"},
            {"name": "ToTensor"}
        ]
    },
    "Voice Conversion": {
        "data_format": "other",
        "data_type": "folder",
        "preprocessing": [
            {"name": "Resample"},
            {"name": "MelSpectrogram"},
            {"name": "ToTensor"}
        ]
    }
}
