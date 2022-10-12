import argparse
import os

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.interpolate import RegularGridInterpolator

from queue import Empty, Queue
import threading
from threading import Thread
from random import randint
from tqdm import tqdm

from libs import mytransforms
from libs.standartminmaxscaler import *
from libs.lstms import UnifiedLSTM
from libs.stationsdata import stations_data
from libs.trig_math import proj_to_trig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate"""

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while not tokill():
        for _, (gfs_batch_tensor, station_batch_tensor, keys) in enumerate(dataset_generator):
            batches_queue.put((gfs_batch_tensor, station_batch_tensor, keys), block=True)
            if tokill():
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, sequence_length):
    while not tokill():
        gfs_batch_tensor, station_batch_tensor, keys = batches_queue.get(block=True)

        # put tensors to cuda
        gfs_batch_tensor = Variable(gfs_batch_tensor).cuda()
        station_batch_tensor = Variable(station_batch_tensor).cuda()

        # divide train and true stations
        train_station_seq = station_batch_tensor[:, :sequence_length]
        y_fcr = station_batch_tensor[:, sequence_length:]

        cuda_batches_queue.put((gfs_batch_tensor, train_station_seq, y_fcr, keys), block=True)
        #         print('i put smth in cuda queue')
        if tokill():
            return


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_objects_i(objects_count):
    """Cyclic generator of object indices
    """
    current_objects_id = 0
    while True:
        yield randint(0, objects_count)


class MPWindDataset(Dataset):
    def __init__(self, gfs_files, station_dir, sequence_length, forecast_range, batch_size=64,
                 station_transform=None, gfs_transform=None):
        print('Init called!')
        gfs_field = []

        for pressure_level in tqdm([0, 1, 2, 7], total=4):
            gfs = []
            for wind_dir in range(2):
                gfs.append(np.load(gfs_files[pressure_level * 2 + wind_dir]).astype(np.float32))
            gfs_field.append(np.stack(proj_to_trig(gfs[0], gfs[1]), axis=1))
        for other_params in tqdm([16, 17], total=2):
            gfs_field.append(np.expand_dims(
                np.load(gfs_files[other_params]).astype(np.float32), axis=1))
        self.ground_wind = [9, 10, 11]
        gfs_field = np.concatenate(gfs_field, axis=1)
        gfs_field = torch.from_numpy(gfs_field)
        self.channels_number = gfs_field.shape[1]
        print(self.channels_number, 'channels')

        stations_list = []
        self.keys_list = []
        for key in tqdm(stations_data, total=len(stations_data)):
            stations_list.append(np.load(os.path.join(station_dir, stations_data[key]['filename'])))
            self.keys_list.append(key)
        stations = np.stack(stations_list, axis=0)

        stations = torch.from_numpy(stations)
        self.targetss = MyStandartScaler()
        self.gfsss = MyStandartScaler()
        self.gfsss.channel_fit(gfs_field, channels=[0, 3, 6, 9], channels_dim=1)
        self.targetss.channel_fit(stations, channels=[0], channels_dim=1)

        self.stations = stations
        self.gfs_field = gfs_field
        self.sequence_length = sequence_length
        self.forecast_range = forecast_range
        self.gfs_transform = gfs_transform
        self.station_transform = station_transform

        self.batch_size = batch_size

        self.objects_id_generator = threadsafe_iter(get_objects_i(len(self)))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.resizer = transforms.Resize((224, 224))
        self.piler = transforms.ToPILImage()
        self.tenser = transforms.ToTensor()
        self.mm = MyMinMaxScaler()

    def __len__(self):
        return self.gfs_field.shape[0] - self.sequence_length - self.forecast_range

    def __iter__(self):
        while True:
            with self.lock:
                gfs_seqs = []
                station_seqs = []
                keys_seq = []
            for idx in self.objects_id_generator:
                gfs_seq = self.gfs_field[idx:idx + self.sequence_length + self.forecast_range]
                st_id = randint(0, len(stations_data) - 1)
                station_seq = self.stations[st_id, idx:idx + self.sequence_length + self.forecast_range]
                if self.station_transform:
                    station_seq[:-self.forecast_range] = self.station_transform(station_seq[:-self.forecast_range])
                if self.gfs_transform:
                    for i in range(len(gfs_seq)):
                        gfs_seq[i] = self.gfs_transform(gfs_seq[i])
                gfs_seq = self.gfsss.channel_transform(gfs_seq)
                station_seq = self.targetss.channel_transform(station_seq, 1)
                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(gfs_seqs)) < self.batch_size:
                        gfs_seqs.append(gfs_seq)
                        station_seqs.append(station_seq)
                        keys_seq.append(self.keys_list[st_id])
                    if len(gfs_seqs) % self.batch_size == 0:
                        gfs_batch_tensor = torch.stack(gfs_seqs)
                        station_batch_tensor = torch.stack(station_seqs)
                        yield gfs_batch_tensor, station_batch_tensor, keys_seq
                        gfs_seqs, station_seqs, keys_seq = [], [], []


def train_epoch(lstm: nn.Module,
                optimizer: torch.optim.Optimizer,
                cuda_batches_queue: Queue,
                criterion: callable,
                dataset: MPWindDataset,
                epoch: int,
                steps_per_epoch: int,
                interpolator_linspace,
                args):
    lstm.train()
    train_loss_avg = 0
    batch_averaged_loss = []
    num_batches = 0
    print(steps_per_epoch)
    for batch_idx in tqdm(range(steps_per_epoch), total=steps_per_epoch):

        gfs_seq, station_seq, target, keys = cuda_batches_queue.get(block=True)
        y = lstm(gfs_seq, station_seq)
        y = y.double()

        # prepare gfs forecast
        gfs_seq = gfs_seq.clone()
        for i in range(len(gfs_seq)):
            gfs_seq[i] = dataset.gfsss.channel_inverse_transform(gfs_seq[i])
            target[i] = dataset.targetss.channel_inverse_transform(target[i])
        target = target.double()
        gfs_forecast = gfs_seq[:, -6:, [9, 10, 11], 35:45, 35:45].cpu().detach().numpy()
        # decided to interpolate angle so convert sin/cos to angle
        gfs_angle = np.arctan2(gfs_forecast[:, :, 1], gfs_forecast[:, :, 2])
        # unwrap angles
        # gfs_angle = gfs_angle.cpu()
        for i, batch in enumerate(gfs_angle):
            for j, hour in enumerate(batch):
                orig_shape = hour.shape
                hour = hour.reshape(-1)
                hour = np.unwrap(hour)
                gfs_angle[i, j] = hour.reshape(orig_shape)
        gfs_forecast = np.stack((gfs_forecast[:, :, 0], gfs_angle), axis=2)
        gfs_interpolator = RegularGridInterpolator(interpolator_linspace, gfs_forecast)
        station_coord = [stations_data[keys[i]]['coord'] for i in range(len(keys))]
        station_points = np.ones((args.batch_size, args.forecast_range, 2, 5))
        for b in range(args.batch_size):
            for t in range(args.forecast_range):
                for p in range(2):
                    station_points[b, t, p] = np.array([b, t, p] + station_coord[b])
        gfs_forecast = gfs_interpolator(station_points)
        # convert back to sin/cos
        gfs_forecast = np.stack((gfs_forecast[:, :, 0], np.sin(gfs_forecast[:, :, 1]), np.cos(gfs_forecast[:, :, 1])),
                                axis=2)
        forecast = torch.from_numpy(gfs_forecast).cuda()
        # calculate true sin/cos forecast correction. if we correct persistence, forecast output is 0h
        true_corr = torch.zeros_like(y)
        for j in range(6):
            # sin(true_corr) = sin(6h)cos(0h)-cos(6h)sin(0h)
            true_corr[:, j, 1] = target[:, j, 1] * forecast[:, j, 2] - target[:, j, 2] * forecast[:, j, 1]
            # cos(true_corr) = cos(6h)cos(0h)-sin(6h)sin(0h)
            true_corr[:, j, 2] = target[:, j, 2] * forecast[:, j, 2] + target[:, j, 1] * forecast[:, j, 1]
        true_corr[:, :, 0] = target[:, :, 0] - forecast[:, :, 0]

        loss = criterion(y, true_corr.cuda())
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optimizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg += loss.item()
        num_batches += 1

        batch_averaged_loss.append(loss.item())
        if batch_idx >= steps_per_epoch - 1:
            break
    return


def main(args):
    # cuda = True if torch.cuda.is_available() else False
    # if cuda:
    #     torch.cuda.set_device(0)
    # cuda_dev = torch.device('cuda:0')
    # for GFS data cropped around Falconara station
    lats = np.linspace(33.75, 53.5, 80)
    lons = np.linspace(3.5, 23.25, 80)
    lats, lons = lats[35:45], lons[35:45]
    # for wind speed and direction
    param_linspace = np.linspace(0, 1, 2)
    # for each hour of forecast
    time_linspace = np.linspace(0, args.forecast_range - 1, args.forecast_range)
    # for each item in batch
    batch_linspace = np.linspace(0, args.batch_size - 1, args.batch_size)

    interpolator_linspace = (batch_linspace, time_linspace, param_linspace, lats, lons)

    station_augmentation = [
        transforms.RandomApply([mytransforms.StationNormalNoize(0, 0.2), ], p=1),
    ]
    gfs_augmentation = [
        transforms.RandomApply([transforms.RandomRotation(3)], p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    ]
    station_augmentation = transforms.Compose(station_augmentation)
    gfs_augmentation = transforms.Compose(gfs_augmentation)

    dataset = MPWindDataset(args.gfs_field_files, args.target_dir, args.sequence_length,
                            args.forecast_range, args.batch_size, station_augmentation, gfs_augmentation)
    steps_per_epoch = len(dataset) // args.batch_size + 1
    batches_queue_length = min(steps_per_epoch, 16)

    print('creating the model')
    lstm = UnifiedLSTM(args.input_size, args.station_params_number, args.hidden_size, args.num_layers,
                       dataset.channels_number, args.forecast_range, dataset.targetss)  # our lstm class
    lstm.cuda()
    print('lstm created')
    # if args.start_epoch != 1:
    #     lstm.load_state_dict(torch.load('epochs/netG_epoch_%d.pth' % (args.start_epoch - 1)))
    #     lstm.load_state_dict(torch.load('epochs/netD_epoch_%d.pth' % (args.start_epoch - 1)))
    # else:
    #     pass

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    criterion.cuda()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=batches_queue_length)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    for _ in range(args.num_workers):
        thr = Thread(target=threaded_batches_feeder, args=(train_thread_killer, train_batches_queue, dataset))
        thr.start()

    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches, args=(
        train_cuda_transfers_thread_killer, train_cuda_batches_queue, train_batches_queue, args.sequence_length))
    train_cudathread.start()

    print('\n\nstart training')
    # start_epoch = 1 if resume_state is None else resume_state.epoch
    for epoch in range(args.start_epoch, args.epochs + 1):
        # print('\n\n%s: Train epoch: %d of %d' % (args.run_name, epoch, EPOCHS))

        print('Train epoch: %d of %d' % (epoch, args.epochs))
        train_epoch(lstm, optimizer, train_cuda_batches_queue, criterion, dataset, epoch, steps_per_epoch,
                    interpolator_linspace, args)

        torch.save(lstm.state_dict(), 'epochs/unilstm_epoch_%d.pth' % epoch)

        scheduler.step(epoch=epoch)

    train_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(args.num_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
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
    parser.add_argument('--target_dir', type=str,
                        default='/app/Windy/Station/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    main(args)
