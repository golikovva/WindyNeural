model_type = "clusterresnet"

model = dict(
    fea_model_type='WideResNet',
    fea_model=dict(
        in_channels=14,
        depth=28,
        widen_factor=10,
        bn_momentum=0.01,
        leaky_slope=0.0,
        dropRate=0.0
    ),
    lstm=dict(
        # gfs_input_size=512,
        station_input_size=5,
        station_output_size=3,
        # gfs_hidden_size=509,
        num_layers=1,
        # in_channels=14,
        forecast_range=6,
    ),
)
