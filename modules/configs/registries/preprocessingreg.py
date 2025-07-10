AUDIO_TRANSFORMS = {
    "Speed": {"orig_freq": 16000, "factor": 1.0},
    "AmplitudeToDB": {"stype": "power", "top_db": None},
    "Resample": {
        "orig_freq": 16000, "new_freq": 16000,
        "resampling_method": "sinc_interp_hann", "lowpass_filter_width": 6,
        "rolloff": 0.99, "beta": None, "dtype": None
    },
    "Fade": {"fade_in_len": 0, "fade_out_len": 0, "fade_shape": "linear"},
    "Vol": {"gain": 1.0, "gain_type": "amplitude"},
    "Loudness": {"sample_rate": 16000},
    "AddNoise": {"snr": 10.0, "lengths": None},

    "Spectrogram": {
        "n_fft": 400, "win_length": None, "hop_length": None, "pad": 0,
        "power": 2, "normalized": False, "center": True,
        "pad_mode": "reflect", "onesided": True
    },
    "MelSpectrogram": {
        "sample_rate": 16000, "n_fft": 400, "n_mels": 120,
        "f_min": 0.0, "f_max": None, "hop_length": None
    },
    "MFCC": {
        "sample_rate": 16000, "n_mfcc": 40, "dct_type": 2,
        "norm": "ortho", "log_mels": False
    },
    "TimeStretch": {"n_freq": 201, "fixed_rate": None},
    "FrequencyMasking": {"freq_mask_param": 30, "iid_masks": False},
    "TimeMasking": {"time_mask_param": 40, "iid_masks": False, "p": 1.0},
}

TEXT_TRANSFORMS = {
    "RegexTokenizer": {"patterns_list": r"\w+"},
    "SentencePieceTokenizer": {"sp_model_path": "model_path.model"},
    "VocabTransform": {"vocab": []},
    "ToTensor": {"dtype": "int64"},
    "Truncate": {"max_seq_len": 128},
    "PadTransform": {"max_length": 128, "pad_value": 0},
    "AddToken": {"token": "<CLS>", "begin": True},
    "BERTTokenizer": {"tokenizer": "facebook/bart-base"},
    "LabelToIndex": {"label_names": []}
}

IMAGE_TRANSFORMS = {
    "Resize": {"size": [224, 224], "interpolation": "bilinear"},
    "RandomCrop": {"size": [224, 224], "padding": None, "pad_if_needed": False},
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 15, "interpolation": "NEAREST"},
    "ColorJitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
    "Grayscale": {"num_output_channels": 1},
    "RandomAdjustSharpness": {"sharpness_factor": 2, "p": 0.5},
    "Normalize": {"mean": [0.5], "std": [0.5]},
    "ConvertImageDtype": {"dtype": "float32"},
    "ToTensor": {},
    "RandomErasing": {"p": 0.5, "scale": [0.02, 0.33], "ratio": [0.3, 3.3], "value": 0},
    "GaussianBlur": {"kernel_size": 3, "sigma": [0.1, 2.0]},
}
