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
from model.models import WeatherLSTM
from libs.mydatasets import WindDataset
from config.stationsdata import stations_data
from libs.trig_math import proj_to_trig
from libs.eval_metrics import *
from main import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(args):
    cfg = Config.fromfile(args.config_file)
    dataset = MPWindDataset(args.val_gfs_files, args.gfs_forecast_files, args.target_dir, args.sequence_length,
                            args.forecast_range, args.batch_size,
                            test_mode=True)
    test_steps = len(dataset) // args.batch_size + 1
    batches_queue_length = min(test_steps, 16)
    model = build_model(args.model_type, cfg.model)
    state_dict = torch.load('./epochs_transformer/transformer_epoch_106.pth')
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
            # target = dataset.targetss.channel_inverse_transform(target, channels_dim=2)
            # target = dataset.targetss.channel_inverse_transform(target, 2)
            loss = criterion(y, true_forecast)
            val_loss_avg += loss.item()

            interpolator = dataset.interpolators[0]
            xidx = interpolator.idxx
            yidx = interpolator.idxy
            # print(gfs_seq.shape, 'gfs_Seq shape before inverse transform')
            # print(gfs_seq.min(), gfs_seq.max(), 'min max before transform')
            print(gfs_seq.shape)
            gfs_seq = dataset.gfsmm.channel_inverse_transform(gfs_seq, channels_dim=2)
            # print(gfs_seq.min(), gfs_seq.max(), 'min max after transform')
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

            show_images = True
            if show_images:
                for h in range(6):
                    plt.figure(figsize=(4, 3), dpi=300)
                    plt.plot(predict[:, h, 0].cpu().detach(), label='pred Data')  # actual plot
                    plt.plot(forecast[:, h, 0].cpu().detach(), label='gfs forecast')  # actual plot
                    plt.plot(target[:, h, 0].cpu().detach(), label='target Data')  # predicted plot
                    plt.plot(y[:, h, 0].cpu().detach(), label='corr')
                    plt.title('Time-Series Prediction')
                    plt.legend()
                    # plt.show()
                    plt.savefig(f'images/result_{h}h_{batch_idx}idx')
                batch_idx += 1
                if batch_idx > 15:
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
    parser.add_argument('--gfs_forecast_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param{i}of18_15011500-22042412_006.npy" for i in
                                 range(18)])
    parser.add_argument('--val_gfs_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_test.npy" for i in
                                 range(18)])
    parser.add_argument('--target_dir', type=str,
                        default='/app/Windy/Station/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='lstm')

    args = parser.parse_args()
    test(args)
