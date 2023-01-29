from .models import *
from .wrn import build_WideResNet
import torchvision


def build_model(model_type, model_cfg, **kwargs):
    print(kwargs)
    if model_type == "transformer":
        print('You have chosen transformer model!')
        return WindTransformer(**model_cfg.transformer)
    elif model_type == "lstm":
        print('You have chosen LSTM model!')
        # try:
        #     kwargs['station_scaler']
        # except KeyError:
        #     raise Exception('LSTM model needs station scaler object to operate!')

        if model_cfg.fea_model_type in ['WideResNet', 'ResNet-18']:
            fea_builder = build_feature_model(model_cfg.fea_model_type,
                                              model_cfg.fea_model)
        else:
            raise TypeError
        return WeatherLSTM(fea_builder, **model_cfg.lstm, **kwargs)
    else:
        raise TypeError


def build_feature_model(model_type, fea_model_cfg):
    if model_type == "ResNet-18":
        print('You have chosen ResNet-18 feature model!')
        rn_builder = build_ResNet(fea_model_cfg)
        return rn_builder.build
    elif model_type == "WideResNet":
        print('You have chosen WideResNet feature model!')
        wrn_builder = build_WideResNet(fea_model_cfg)
        return wrn_builder.build
    else:
        raise TypeError
