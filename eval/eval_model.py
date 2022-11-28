import argparse
import os
import sys

import torch

sys.path.insert(0, './')

from torch.utils.data import DataLoader
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt

from tqdm import tqdm

# from libs import mytransforms
from libs.standartminmaxscaler import *
from model.models import UnifiedLSTM
from libs.mydatasets import WindDataset
from config.stationsdata import stations_data
from libs.trig_math import proj_to_trig
from libs.eval_metrics import *
from main import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(args):
    cfg = Config.fromfile(args.config_file)
    dataset = MPWindDataset(args.val_gfs_files, args.target_dir, args.sequence_length,
                            args.forecast_range, args.batch_size,
                            test_mode=True)
    test_steps = len(dataset) // args.batch_size + 1
    batches_queue_length = min(test_steps, 16)
    model = build_model(args.model_type, cfg.model, station_scaler=dataset.targetss)
    state_dict = torch.load('./epochs/unilstm_best_epoch_88.pth')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    criterion.cuda()
    val_batches_queue = Queue(maxsize=batches_queue_length)
    val_cuda_batches_queue = Queue(maxsize=4)
    val_thread_killer = thread_killer()
    val_thread_killer.set_tokill(False)

    for _ in range(args.num_workers):
        thr = Thread(target=threaded_batches_feeder, args=(val_thread_killer, val_batches_queue, dataset))
        thr.start()
    val_cuda_transfers_thread_killer = thread_killer()
    val_cuda_transfers_thread_killer.set_tokill(False)
    val_cudathread = Thread(target=threaded_cuda_batches, args=(
        val_cuda_transfers_thread_killer, val_cuda_batches_queue, val_batches_queue, args.sequence_length))
    val_cudathread.start()

    model.eval()
    with torch.no_grad():
        val_loss_avg = 0
        forecast_stat = []
        gfs_forecast_stat = []
        target_stat = []
        for batch_idx in tqdm(range(test_steps), total=test_steps):
            gfs_seq, station_seq, target, true_forecast = val_cuda_batches_queue.get(block=True)
            y = model(gfs_seq, station_seq)
            target = dataset.targetss.channel_inverse_transform(target, channels_dim=2)
            # target = dataset.targetss.channel_inverse_transform(target, 2)
            loss = criterion(y, true_forecast)
            val_loss_avg += loss.item()

            interpolator = dataset.interpolators[0]
            xidx = interpolator.idxx
            yidx = interpolator.idxy
            gfs_seq = dataset.gfsss.channel_inverse_transform(gfs_seq, channels_dim=2)

            forecast = gfs_seq[:, -6:, dataset.ground_wind, xidx:xidx+2, yidx:yidx+2]
            forecast = dataset.interpolators[0].interpolate(forecast)

            predict = torch.zeros_like(y)
            for j in range(6):
                # sin(pred) = cos(corr)sin(gfs)+cos(gfs)sin(corr)
                predict[:, j, 1] = forecast[:, j, 1] * y[:, j, 2] + forecast[:, j, 2] * y[:, j, 1]
                # cos(pred) = cos(gfs)cos(corr)-sin(gfs)sin(corr)
                predict[:, j, 2] = y[:, j, 2] * forecast[:, j, 2] - y[:, j, 1] * forecast[:, j, 1]
            predict[:, :, 0] = y[:, :, 0] + forecast[:, :, 0]
            forecast_stat.append(predict[:, :, 0].cpu().detach())
            gfs_forecast_stat.append((forecast[:, :, 0].cpu().detach()))
            target_stat.append(target[:, :, 0].cpu().detach())

            show_images = False
            if show_images:
                plt.figure(figsize=(4, 3), dpi=100)
                plt.plot(predict[:, 5, 0].cpu().detach(), label='pred Data')  # actual plot
                plt.plot(forecast[:, 5, 0].cpu().detach(), label='gfs forecast')  # actual plot
                plt.plot(target[:, 5, 0].cpu().detach(), label='target Data')  # predicted plot
                plt.plot(y[:, 5, 0].cpu().detach(), label='corr')
                plt.title('Time-Series Prediction')
                plt.legend()
                # plt.show()
                plt.savefig(f'result_5h_{batch_idx}idx')
                batch_idx += 1
                if batch_idx > 5:
                    break

        avg_loss = val_loss_avg / test_steps
        losses_dict = {'avg_loss': avg_loss}
        print('LSTM metrics')
        forecast = np.concatenate(forecast_stat, 0)
        target = np.concatenate(target_stat, 0)
        # print(forecast[:60, 5], target[:60, 5])
        calc_wRMSE(forecast, target)
        forecast = np.concatenate(forecast_stat, 0)
        target = np.concatenate(target_stat, 0)
        calc_cat_change_metric(forecast, target)
        print('GFS metrics')
        forecast = np.concatenate(gfs_forecast_stat, 0)
        target = np.concatenate(target_stat, 0)
        # print(forecast[:60, 5], target[:60, 5])
        calc_wRMSE(forecast, target)
        forecast = np.concatenate(gfs_forecast_stat, 0)
        target = np.concatenate(target_stat, 0)
        calc_cat_change_metric(forecast, target)

        val_thread_killer.set_tokill(True)
        val_cuda_transfers_thread_killer.set_tokill(True)
        for _ in range(args.num_workers):
            try:
                # Enforcing thread shutdown
                val_batches_queue.get(block=True, timeout=1)
                val_cuda_batches_queue.get(block=True, timeout=1)
            except Empty:
                pass
        return losses_dict


def main(args):
    print(os.path.abspath(__file__))
    gfs_files = [f"/app/Windy/Station/GFS_falconara_param{i}of18_15011500-22042412.npy" for i in range(18)]
    station_dir = '/app/Windy/Station/'
    gfs_field = []
    for pressure_level in tqdm([0, 1, 2, 7], total=4):
        gfs = []
        for wind_dir in range(2):
            gfs.append(np.load(gfs_files[pressure_level * 2 + wind_dir]).astype(np.float32))
        gfs_field.append(np.stack(proj_to_trig(gfs[0], gfs[1]), axis=1))
    for other_params in tqdm([16, 17], total=2):
        gfs_field.append(np.expand_dims(
            np.load(gfs_files[other_params]).astype(np.float32), axis=1))
    ground_wind = [9, 10, 11]
    gfs_field = np.concatenate(gfs_field, axis=1)
    gfs_field = torch.from_numpy(gfs_field)
    channels_number = gfs_field.shape[1]
    print(channels_number, 'channels')

    stations_list = []
    keys_list = []
    for key in tqdm(stations_data, total=len(stations_data)):
        if key == 'falconara':
            stations_list.append(np.load(os.path.join(station_dir, stations_data[key]['filename'])))
            keys_list.append(key)
    stations = np.stack(stations_list, axis=0)
    stations = np.squeeze(stations)
    stations = torch.from_numpy(stations)
    print(stations.shape, 'stations.shape')
    print(gfs_field.shape, 'gfs.shape')
    print(stations[:12], 'station')
    print(gfs_field[1], 'gfs')
    targetss = MyStandartScaler()
    gfsss = MyStandartScaler()
    gfsss.channel_fit_transform(gfs_field, channels=[0, 3, 6, 9], channels_dim=1)
    targetss.channel_fit_transform(stations, channels=[0], channels_dim=1)
    print(stations[:12], 'station 1')
    print(gfs_field[1], 'gfs 1')
    test_data = gfs_field[60000:, :]
    test_target = stations[60000:, :]
    data = gfs_field[:60000, :]
    target = stations[:60000, :]
    forecast_range = 6

    lstm = UnifiedLSTM(args.input_size, args.station_params_number, args.hidden_size, args.num_layers,
                       14, args.forecast_range, targetss)
    lstm.load_state_dict(torch.load('./epochs/unilstm_epoch_10.pth'))
    lstm.eval()
    lstm.to(device)

    test_dataset = WindDataset(test_data, test_target, 72, forecast_range, device)
    batch_size = args.batch_size
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lats = np.linspace(33.75, 53.5, 80)
    lons = np.linspace(3.5, 23.25, 80)
    lats, lons = lats[39:41], lons[39:41]
    # for wind speed and direction
    param_linspace = np.linspace(0, 1, 2)
    # for each hour of forecast
    time_linspace = np.linspace(0, forecast_range - 1, forecast_range)
    # for each item in batch
    batch_linspace = np.linspace(0, batch_size - 1, batch_size)

    falconara_coord = [43.61, 13.36]
    falconara_points = np.ones((batch_size, forecast_range, 2, 5))
    for b in range(batch_size):
        for t in range(forecast_range):
            for p in range(2):
                falconara_points[b, t, p] = np.array([b, t, p] + falconara_coord)

    lstm.eval()
    eval_loss_avg = []
    print('Evaluating ...')
    plt.figure(figsize=(8, 6), dpi=300)
    forecast_stat = []
    target_stat = []
    k = 0
    for era_data, station_data, target in test_dataloader:

        corr = lstm(era_data, station_data)
        corr = corr.cpu().detach().numpy()
        print(era_data.shape, 'eradata shape before transform')
        print(target.shape, 'tagret shape before transform')
        era_data = gfsss.channel_inverse_transform(era_data, 2).cpu().detach().numpy()
        target = targetss.channel_inverse_transform(target, 2).cpu().detach()

        # prepare gfs forecast
        gfs_forecast = era_data[:, -6:, ground_wind, 39:41, 39:41]
        print(gfs_forecast[1], 'gfs in loop')
        print(target[:, 1, 0], 'target in loop')
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
        GFS_interpolator = RegularGridInterpolator((batch_linspace, time_linspace, param_linspace, lats, lons),
                                                   gfs_forecast)
        gfs_forecast = GFS_interpolator(falconara_points)
        # convert back to sin/cos
        gfs_forecast = np.stack((gfs_forecast[:, :, 0], np.sin(gfs_forecast[:, :, 1]), np.cos(gfs_forecast[:, :, 1])),
                                axis=2)
        forecast = gfs_forecast
        print(forecast.shape, 'forecast shape')

        # calculate true sin/cos forecast correction. if we correct persistence, forecast output is 0h
        predict = np.zeros_like(corr)
        for j in range(6):
            # sin(pred) = cos(corr)sin(gfs)+cos(gfs)sin(corr)
            predict[:, j, 1] = forecast[:, j, 1] * corr[:, j, 2] + forecast[:, j, 2] * corr[:, j, 1]
            # cos(pred) = cos(gfs)cos(corr)-sin(gfs)sin(corr)
            predict[:, j, 2] = corr[:, j, 2] * forecast[:, j, 2] - corr[:, j, 1] * forecast[:, j, 1]
        # predict[:, :, 0] = corr[:, :, 0] + forecast[:, :, 0]
        predict[:, :, 0] = forecast[:, :, 0]
        print(corr[:50, :, 0])

        forecast_stat.append(predict[:, :, 0])
        target_stat.append(target[:, :, 0])
        #
        plt.figure(figsize=(4, 3), dpi=100)
        plt.plot(predict[:, 5, 0], label='pred Data')  # actual plot
        plt.plot(target[:, 5, 0], label='target Data')  # predicted plot
        plt.plot(corr[:, 5, 0], label='corr')
        #     print(target[:, 5, 2], pred[:, 5, 0])
        plt.title('Time-Series Prediction')
        plt.legend()
        # plt.show()
        plt.savefig(f'result_5h_{k}idx')
        k += 1
        if k > 5:
            break

    forecast = np.concatenate(forecast_stat, 0)
    target = np.concatenate(target_stat, 0)
    # print(forecast[:60, 5], target[:60, 5])
    calc_wRMSE(forecast, target)
    forecast = np.concatenate(forecast_stat, 0)
    target = np.concatenate(target_stat, 0)
    calc_cat_change_metric(forecast, target)
    # forecast = torch.cat(forecast_stat, 0)
    # target = torch.cat(target_stat, 0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--config-file",
        default="./train_config.py",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--save_name', type=str, default='unified_lstm')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_false')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='total number of batch size of labeled data')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--forecast_range', type=int, default=6)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=509)
    parser.add_argument('--station_params_number', type=int, default=3)
    parser.add_argument('--sequence_length', type=int, default=72)
    parser.add_argument('--gfs_field_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param{i}of18_15011500-22042412.npy" for i in
                                 range(18)])
    parser.add_argument('--val_gfs_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_test.npy" for i in
                                 range(18)])
    parser.add_argument('--target_dir', type=str,
                        default='/app/Windy/Station/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='lstm')

    args = parser.parse_args()
    test(args)
