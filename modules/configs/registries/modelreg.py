PRETRAINED_MODEL_REGISTRY = {
    "ResNet18": {"pretrained": True, "num_classes": 1000},
    "ResNet50": {"pretrained": True, "num_classes": 1000},
    "EfficientNetB0": {"width_mult": 1.0, "depth_mult": 1.0, "dropout": 0.2},
    "EfficientNetB7": {"width_mult": 2.0, "depth_mult": 3.1, "dropout": 0.5},
    "ViT": {"image_size": 224, "patch_size": 16, "num_layers": 12,
            "num_heads": 12, "hidden_dim": 768},

    "FasterRCNN": {"backbone": "resnet50", "num_classes": 91,
                   "min_size": 800, "max_size": 1333},
    "MaskRCNN": {"backbone": "resnet50", "num_classes": 91,
                 "box_detections_per_img": 100},

    "DeepLabV3": {"backbone": "resnet50", "atrous_rates": (6, 12, 18),
                  "num_classes": 21},

    "Conformer": {"input_dim": 80, "num_heads": 4, "ffn_dim": 256,
                  "num_layers": 6, "depthwise_conv_kernel_size": 31,
                  "dropout": 0.0, "use_group_norm": False, "convolution_first": False},

    "Wave2Letter": {"num_classes": 40, "input_type": "waveform", "num_features": 1},    

    "WaveRNN": {"upsample_scales": [5, 5, 8], "n_classes": 256, "hop_length": 200,
                "n_res_block": 10, "n_rnn": 512, "n_fc": 512, "kernel_size": 5,
                "n_freq": 128, "n_hidden": 128, "n_output": 128},

    "GloVe": {"dim": 300, "name": "6B"},
    "FastText": {"language": "en"},

    "TransformerEncoderDecoder": {"d_model": 512, "nhead": 8,
                                  "num_encoder_layers": 6, "num_decoder_layers": 6},

    "XLMRoberta": {"num_classes": 2, "dropout": 0.1, "pooler_type": "cls"}
}
