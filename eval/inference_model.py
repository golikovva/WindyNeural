import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.interpolate import RegularGridInterpolator

import sys

sys.path.insert(0, './')

from libs.standartminmaxscaler import *
from libs.lstms import *
from libs.trig_math import *
from libs.stationsdata import *


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference_model(gfs_field, station_seq, model_path, wind_forecast_channels, station_coords=None, lat_bounds=None,
                    lon_bounds=None):
    '''data required shape: gfs - station_number x sequence_length+forecast_range x channels number x h x w
                            station - station_number x sequence_length x 3
                            for station 3 values expected: wind speed module, sin(alpha), cos(alpha),
                            where alpha means angle measured in clockwise direction between true north
                            and the direction from which the wind is blowing.
    '''
    '''gfs_field - поле значений gfs вокруг станции. размер поля h на w задается из геофизических соображений. Например,
    для данных ERA5 было выбрано h=w=72, что эквивалентно 72/4=18 градусам - характерный размер мезомасштабных
    процессов. В зависимости от выбранной нейросетевой модели количество необходимых данных может меняться. Например,
    могут нейросети могут быть необходимы данные о скорости ветра, синусе и косинусе его направления на нескольких
    изобарических высотах, давлении на уровне моря, и температуре воздуха у поверхности. 
    В сумме 3 * кол-во высот + 2 канала данных (обозначим channels). 
    Итого, размер тензора gfs поля в определеный момент времени channels x h x w.
    Нейросеть строит свой прогноз как коррекцию прогноза gfs. для этого ей в параметре wind_forecast_channels нужно
    передать список порядковых номеров каналов gfs со скоростью и тригонометрией ветра наиболее коррелирующими
    с показаниями станции.
    Например для следующих параметров gfs...
    parameters_gfs = [
        'levels_250_speed',
        'levels_250_sin',
        'levels_250_cos',
        'levels_500_speed',
        'levels_500_sin',
        'levels_500_cos',
        'levels_1000_cos',
        'levels_1000_sin',
        'levels_1000_cos',
        'PRSML',
        'TMP'
    ]
    нужно указать wind_forecast_channels = [6, 7, 8] - индексы данных прогноза ветра на высоте 1000gpa
    station_seq - измерение модуля скорости ветра (скаляр) и тригонометрии его направления на станции
    за последние 72 часа
    
    Подразумевается, что данные нескольких станций могут обрабатываться одной моделью.
    Их количество обозначим - station_number
    Нейросети требуются данные ветра со станции и gfs за последние 72 часа. Для разных моделей эта цифра может
    отличаться, поэтому обозначим её как sequence_length
    Также необходим проноз численной модели gfs на 6 часов вперед. Дальность прогноза обозначим forecast_range
    Таким образом требуемые размеры входных тензоров:
                            gfs - station_number x sequence_length+forecast_range x channels x h x w
                            station - station_number x sequence_length x 3
                            station_number - количество станций, с которых собираются данные для модели
                            3 - модуль скорости ветра на станции, sin(alpha), cos(alpha), где alpha - угол,
                            измеренный против часовой стрелки, между направлением
                            на север и направлением откуда дует ветер
                            
    необязательные параметры station_coords, lat_bounds, lon_bounds нужны для интерполяции поля gfs в координату станции
    station_coords - список из координат станций - списков из 2 значений
    lat_bounds - список из 3 значений - южная и северная широты ограничивающие домен gfs и h размер домена по высоте. 
    [min lat, max lat, h]
    lon_bounds - список из 3 значений - восточная и западная долготы ограничивающие домен gfs и w размер домена по ширине
    [min lon, max lon, w]
     Пример использования функции:
        gfs_field = np.load(f"./GFS_falconara_15011500-22042412_14param_test.npy")
        gfs_field = torch.from_numpy(gfs_field)
        stations = np.load(f"./4stations_test.npy")
        stations = torch.from_numpy(stations)
        # gfs_field
        gfs_field = gfs_field[:78]
        gfs_field = torch.stack((gfs_field, gfs_field, gfs_field, gfs_field))
        stations = stations[:, :72]
    
        print(stations.shape)  # torch.Size([4, 72, 3])
        print(gfs_field.shape)  # torch.Size([4, 78, 14, 80, 80])
        station_coords = [[43.61, 13.36], [43.52, 12.73], [43.09, 12.51], [44.02, 12.61]]
        lat_bounds = [33.75, 53.5, 80]
        lon_bounds = [3.5, 23.25, 80]
        wind_forecast_channels = [9, 10, 11]
        pred = inference_model(gfs_field, stations, './epochs/unilstm_epoch_10.pth', wind_forecast_channels
                               , station_coords, lat_bounds, lon_bounds)
    '''
    channels_number = gfs_field.shape[1]
    print(channels_number, 'channels got')
    targetss = MyStandartScaler()
    gfsss = MyStandartScaler()
    gfsss.channel_fit_transform(gfs_field, channels=[0, 3, 6, 9], channels_dim=1)
    targetss.channel_fit_transform(station_seq, channels=[0], channels_dim=2)

    forecast_range = 6
    sequence_length = 72
    input_size = 512
    station_params_number = 3
    gfs_params_number = 14
    hidden_size = 509
    num_layers = 1

    lstm = UnifiedLSTM(input_size, station_params_number, hidden_size, num_layers,
                       gfs_params_number, forecast_range, targetss)
    lstm.load_state_dict(torch.load(model_path))
    lstm.cpu()
    lstm.eval()
    if not lat_bounds or not lon_bounds or not station_coords:
        print('Station or gfs coords are not specified. Assuming station is in the gfs field center...')
        lat_bounds = [0, gfs_field.shape[-1] - 1, gfs_field.shape[-1]]
        lon_bounds = [0, gfs_field.shape[-1] - 1, gfs_field.shape[-1]]
        station_coord = gfs_field.shape[-1]/2
        station_coords = [[station_coord, station_coord] for _ in range(station_seq.shape[0])]

    lats = np.linspace(*lat_bounds)
    lons = np.linspace(*lon_bounds)
    lats, lons = lats[35:45], lons[35:45]
    # for wind speed and direction
    param_linspace = np.linspace(0, 1, 2)
    # for each hour of forecast
    time_linspace = np.linspace(0, forecast_range - 1, forecast_range)
    # for each station
    stations_linspace = np.linspace(0, station_seq.shape[0] - 1, station_seq.shape[0])

    falconara_points = np.ones((station_seq.shape[0], forecast_range, 2, 5))
    for s in range(station_seq.shape[0]):
        for t in range(forecast_range):
            for p in range(2):
                falconara_points[s, t, p] = np.array([s, t, p] + station_coords[s])

    corr = lstm(gfs_field, station_seq)
    corr = corr.cpu().detach().numpy()
    gfs_field = gfsss.channel_inverse_transform(gfs_field, 2)

    # prepare gfs forecast
    gfs_forecast = gfs_field[:, -6:, wind_forecast_channels, 35:45, 35:45].detach().numpy()
    # decided to interpolate angle so convert sin/cos to angle
    gfs_angle = np.arctan2(gfs_forecast[:, :, 1], gfs_forecast[:, :, 2])
    # unwrap angles
    for i, batch in enumerate(gfs_angle):
        for j, hour in enumerate(batch):
            orig_shape = hour.shape
            hour = hour.reshape(-1)
            hour = np.unwrap(hour)
            gfs_angle[i, j] = hour.reshape(orig_shape)
    gfs_forecast = np.stack((gfs_forecast[:, :, 0], gfs_angle), axis=2)
    gfs_interpolator = RegularGridInterpolator((stations_linspace, time_linspace, param_linspace, lats, lons),
                                               gfs_forecast)
    gfs_forecast = gfs_interpolator(falconara_points)
    # convert back to sin/cos
    gfs_forecast = np.stack((gfs_forecast[:, :, 0], np.sin(gfs_forecast[:, :, 1]), np.cos(gfs_forecast[:, :, 1])),
                            axis=2)
    forecast = gfs_forecast

    # calculate true sin/cos forecast correction. in case we correct persistence, forecast output is 0h
    predict = np.zeros_like(corr)
    for j in range(6):
        # sin(pred) = cos(corr)sin(gfs)+cos(gfs)sin(corr)
        predict[:, j, 1] = forecast[:, j, 1] * corr[:, j, 2] + forecast[:, j, 2] * corr[:, j, 1]
        # cos(pred) = cos(gfs)cos(corr)-sin(gfs)sin(corr)
        predict[:, j, 2] = corr[:, j, 2] * forecast[:, j, 2] - corr[:, j, 1] * forecast[:, j, 1]
    predict[:, :, 0] = corr[:, :, 0] + forecast[:, :, 0]
    # predict[:, :, 0] = forecast[:, :, 0]
    return predict


if __name__ == '__main__':
    gfs_field = np.load(f"/app/Windy/Station/GFS_falconara_15011500-22042412_14param_test.npy")
    gfs_field = torch.from_numpy(gfs_field)
    stations = np.load(f"/app/Windy/Station/4stations_test.npy")
    stations = torch.from_numpy(stations)
    # gfs_field
    gfs_field = gfs_field[:78]

    gfs_field = torch.stack((gfs_field, gfs_field, gfs_field, gfs_field))
    stations = stations[:, :72]

    print(stations.shape)  # torch.Size([4, 72, 3])
    print(gfs_field.shape)  # torch.Size([4, 78, 14, 80, 80])
    station_coords = [[43.61, 13.36], [43.52, 12.73], [43.09, 12.51], [44.02, 12.61]]
    lat_bounds = [33.75, 53.5, 80]
    lon_bounds = [3.5, 23.25, 80]
    wind_forecast_channels = [9, 10, 11]
    pred = inference_model(gfs_field, stations, './epochs/unilstm_epoch_10.pth', wind_forecast_channels
                           , station_coords, lat_bounds, lon_bounds)
    print(pred)
