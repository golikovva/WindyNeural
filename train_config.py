model_type = "clusterresnet"

model = dict(
    lstm=dict(
        gfs_input_size=512,
        station_input_size=5,
        station_output_size=3,
        gfs_hidden_size=509,
        num_layers=1,
        in_channels=14,
        forecast_range=6,
    ),
    transformer=dict(
        in_channels=14,
        nhead=16,
        num_encoder_layers=12,
        batch_first=True,
        forecast_range=6,
        sequence_len=72,
        station_output_channels=3,
    ),

)
