LAYER_REGISTRY = {
    "Linear": {"in_features": 128, "out_features": 64, "bias": True},
    "Bilinear": {"in1_features": 128, "in2_features": 128, "out_features": 64, "bias": True},

    "Conv1d": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0,
               "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"},
    "Conv2d": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,
               "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"},
    "Conv3d": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0,
               "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"},

    "ConvTranspose1d": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1,
                        "padding": 0, "output_padding": 0, "bias": True},
    "ConvTranspose2d": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1,
                        "padding": 1, "output_padding": 0, "bias": True},
    "ConvTranspose3d": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "stride": 1,
                        "padding": 0, "output_padding": 0, "bias": True},

    "MaxPool1d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1, "ceil_mode": False},
    "MaxPool2d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1, "ceil_mode": False},
    "MaxPool3d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1, "ceil_mode": False},

    "AvgPool1d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1,
                  "ceil_mode": False, "count_include_pad": True},
    "AvgPool2d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1,
                  "ceil_mode": False, "count_include_pad": True},
    "AvgPool3d": {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1,
                  "ceil_mode": False, "count_include_pad": True},

    "BatchNorm1d": {"num_features": 64, "eps": 1e-5, "momentum": 0.1,
                    "affine": True, "track_running_stats": True},
    "BatchNorm2d": {"num_features": 64, "eps": 1e-5, "momentum": 0.1,
                    "affine": True, "track_running_stats": True},
    "BatchNorm3d": {"num_features": 64, "eps": 1e-5, "momentum": 0.1,
                    "affine": True, "track_running_stats": True},

    "LayerNorm": {"normalized_shape": 128, "eps": 1e-5, "elementwise_affine": True},

    "Transformer": {"d_model": 512, "nhead": 8, "num_encoder_layers": 6,
                    "num_decoder_layers": 6, "dim_feedforward": 2048,
                    "dropout": 0.1, "activation": "relu"},

    "MultiheadAttention": {"embed_dim": 512, "num_heads": 8, "dropout": 0.0,
                           "bias": True, "add_bias_kv": False},

    "Dropout": {"p": 0.5, "inplace": False},
    "Dropout1d": {"p": 0.5, "inplace": False},
    "Dropout2d": {"p": 0.5, "inplace": False},
    "Dropout3d": {"p": 0.5, "inplace": False},

    "Embedding": {"num_embeddings": 10000, "embedding_dim": 300,
                  "padding_idx": None, "max_norm": None, "sparse": False},

    "PixelShuffle": {"upscale_factor": 2},
    "Upsample": {"size": None, "scale_factor": 2.0, "mode": "nearest"},

    "LSTM": {"input_size": 128, "hidden_size": 256, "num_layers": 1,
             "batch_first": False, "bidirectional": False},

    "Flatten": {"start_dim": 1, "end_dim": -1},
    "Unfold": {"kernel_size": 3, "stride": 1, "padding": 0, "dilation": 1}
}