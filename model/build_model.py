from .models import *


def build_model(model_type, model_cfg, **kwargs):
    print(kwargs)
    if model_type == "transformer":
        return WindTransformer(**model_cfg.transformer)
    elif model_type == "lstm":
        try:
            kwargs['station_scaler']
        except KeyError:
            raise Exception('LSTM model needs station scaler object to operate!')
        return UnifiedLSTM(**model_cfg.lstm, **kwargs)
    else:
        raise TypeError
