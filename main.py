import argparse
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from queue import Empty, Queue
import threading
from threading import Thread
from random import randint
from tqdm import tqdm

from libs import mytransforms
from libs.standartminmaxscaler import *
from config.stationsdata import stations_data
from libs.trig_math import proj_to_trig
from libs.interpolatation import *
from model.build_model import *
from config.config import Config

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
        for _, (gfs_batch_tensor, station_batch_tensor, forecast) in enumerate(dataset_generator):
            batches_queue.put((gfs_batch_tensor, station_batch_tensor, forecast), block=True)
            if tokill():
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, sequence_length):
    while not tokill():
        gfs_batch_tensor, station_batch_tensor, forecast = batches_queue.get(block=True)

        # put tensors to cuda
        gfs_batch_tensor = Variable(gfs_batch_tensor).cuda()
        station_batch_tensor = Variable(station_batch_tensor).cuda()
        forecast = Variable(forecast).cuda()

        # divide train and true stations
        train_station_seq = station_batch_tensor[:, :sequence_length]
        y_fcr = station_batch_tensor[:, sequence_length:]

        cuda_batches_queue.put((gfs_batch_tensor, train_station_seq, y_fcr, forecast), block=True)
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
        # yield randint(0, objects_count)
        yield current_objects_id
        current_objects_id = (current_objects_id + 1) % objects_count

class MPWindDataset(Dataset):
    def __init__(self, gfs_files, gfs_forecast_files, station_dir, sequence_length, forecast_range, batch_size=64,
                 station_transform=None, gfs_transform=None, test_mode=False, train=True):
        print('Initializing MPWindDataset!...')
        gfs_field = []
        # gfs_forecast_field = []
        # 4 isobaric height levels - 250, 500, 750, 1000 gpa
        print('Loading GFS wind speed data...')
        for pressure_level in tqdm([0, 1, 2, 3], total=4):
            gfs = []
            # gfs_forecast = []
            for wind_dir in range(2):
                gfs.append(np.load(gfs_files[pressure_level * 2 + wind_dir], mmap_mode='r').astype(np.float32))
                # gfs_forecast.append(np.load(gfs_forecast_files[pressure_level * 2 + wind_dir], mmap_mode='r').astype(np.float32))
            gfs_field.append(np.stack(proj_to_trig(gfs[0], gfs[1]), axis=1))
            # gfs_forecast_field.append(np.stack(proj_to_trig(gfs_forecast[0], gfs_forecast[1]), axis=1))
        print('Loading other GFS data...')
        for other_params in tqdm([8, 9], total=2):
            gfs_field.append(np.expand_dims(
                np.load(gfs_files[other_params]).astype(np.float32), axis=1))
            # gfs_forecast_field.append(np.expand_dims(
            #     np.load(gfs_forecast_files[other_params]).astype(np.float32), axis=1))
        self.ground_wind = [9, 10, 11]
        gfs_field = np.concatenate(gfs_field, axis=1)
        gfs_field = torch.from_numpy(gfs_field)

        # gfs_forecast_field = np.concatenate(gfs_forecast_field, axis=1)
        # gfs_forecast_field = torch.from_numpy(gfs_forecast_field)
        self.channels_number = gfs_field.shape[1]
        print(self.channels_number, 'channels')

        stations_list = []
        self.keys_list = []
        self.interpolators = []
        self.x = np.linspace(33.75, 53.5, 80)
        self.y = np.linspace(3.5, 23.25, 80)
        filename = 'val_file' if test_mode else 'filename'
        print('Loading stations data...')
        for key in tqdm(stations_data, total=len(stations_data)):
            print(stations_data[key][filename])
            stations_list.append(np.load(os.path.join(station_dir, stations_data[key][filename])))
            interpolator = Interpolator(self.x, self.y)
            interpolator.get_bilinear_weights(stations_data[key]['coord'])
            self.interpolators.append(interpolator)
            self.keys_list.append(key)
        stations = np.stack(stations_list, axis=0)
        stations = torch.from_numpy(stations)
        stations = stations.type(torch.float)
        gfs_field = gfs_field.type(torch.float)

        # gfs_forecast_field = gfs_forecast_field.type(torch.float)
        self.test_mode = test_mode

        self.targetmm = MyMinMaxScaler()
        self.gfsmm = MyMinMaxScaler()
        print(gfs_field.shape, stations.shape)
        self.gfsmm.channel_fit(gfs_field, channels_dim=1)
        self.targetmm.channel_fit(stations, channels_dim=2)
        print(self.targetmm.channel_mins, self.targetmm.channel_maxs)
        print(self.gfsmm.channel_mins, self.gfsmm.channel_maxs)
        self.stations = stations
        self.gfs_field = gfs_field
        # self.gfs_forecast_field = gfs_forecast_field
        self.sequence_length = sequence_length
        self.forecast_range = forecast_range
        self.gfs_transform = gfs_transform
        self.station_transform = station_transform

        self.batch_size = batch_size

        self.objects_id_generator = threadsafe_iter(get_objects_i(len(self)))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0
        self.interpolator_linspace = self.get_interpolation_linspace()
        print("INIT COMPLETED!")
        print(self.gfs_field.shape, self.stations.shape)

    def get_interpolation_linspace(self):
        lats = np.linspace(33.75, 53.5, 80)
        lons = np.linspace(3.5, 23.25, 80)
        lats, lons = lats[35:45], lons[35:45]
        # for wind speed and direction
        param_linspace = np.linspace(0, 1, 2)
        # for each hour of forecast
        time_linspace = np.linspace(0, self.forecast_range - 1, self.forecast_range)
        # for each item in batch
        batch_linspace = np.linspace(0, self.batch_size - 1, self.batch_size)
        interpolator_linspace = (time_linspace, param_linspace, lats, lons)
        return interpolator_linspace

    def __len__(self):
        return self.gfs_field.shape[0] - self.sequence_length - self.forecast_range

    def __iter__(self):
        while True:
            with self.lock:
                if self.init_count == 0:
                    gfs_seqs = []
                    station_seqs = []
                    forecasts = []
                    self.init_count = 1
            for idx in self.objects_id_generator:
                # 1) get data
                gfs_seq = self.gfs_field[idx:idx + self.sequence_length + self.forecast_range].clone()
                # gfs_seq = self.gfs_field[idx:idx + self.sequence_length].clone()
                # fc = self.gfs_forecast_field[idx + self.sequence_length:idx + self.sequence_length + self.forecast_range].clone()
                # gfs_seq = torch.cat((gfs_seq, fc), dim=0)
                st_id = randint(0, len(stations_data) - 1)
                station_seq = self.stations[st_id, idx:idx + self.sequence_length + self.forecast_range].clone()
                interpolator = self.interpolators[st_id]
                # 1.1) get gfs forecast in coord of station
                gfs_forecast = gfs_seq[-6:, self.ground_wind,
                                       interpolator.idxx:interpolator.idxx + 2,
                                       interpolator.idxy:interpolator.idxx + 2]
                gfs_forecast = interpolator.interpolate(gfs_forecast)
                # 1.2) calculate true correction of interpolated gfs
                true_corr = torch.zeros_like(gfs_forecast, device=torch.device(device))
                # sin(true_corr) = sin(6h)cos(0h)-cos(6h)sin(0h)
                # print(true_corr.shape, station_seq.shape, gfs_forecast.shape)
                true_corr[:, 1] = station_seq[-self.forecast_range:, 1] * gfs_forecast[:, 2] - station_seq[
                                                                                               -self.forecast_range:,
                                                                                               2] * gfs_forecast[:, 1]
                # cos(true_corr) = cos(6h)cos(0h)-sin(6h)sin(0h)
                true_corr[:, 2] = station_seq[-self.forecast_range:, 2] * gfs_forecast[:, 2] + station_seq[
                                                                                               -self.forecast_range:,
                                                                                               1] * gfs_forecast[:, 1]
                true_corr[:, 0] = station_seq[-self.forecast_range:, 0] - gfs_forecast[:, 0]
                # 2) scale data to [0, 1]
                gfs_seq = self.gfsmm.channel_transform(gfs_seq, 1)
                station_seq = self.targetmm.channel_transform(station_seq, 1)
                # 3) augment data
                if self.station_transform:
                    station_seq[:-self.forecast_range] = self.station_transform(station_seq[:-self.forecast_range])
                if self.gfs_transform:
                    gfs_seq = self.gfs_transform(gfs_seq)
                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(gfs_seqs)) < self.batch_size:
                        gfs_seqs.append(gfs_seq)
                        station_seqs.append(station_seq)
                        forecasts.append(true_corr)
                    if len(gfs_seqs) % self.batch_size == 0:
                        gfs_batch_tensor = torch.stack(gfs_seqs)
                        station_batch_tensor = torch.stack(station_seqs)
                        forecast_batch_tensor = torch.stack(forecasts)
                        yield gfs_batch_tensor, station_batch_tensor, forecast_batch_tensor

                        gfs_seqs, station_seqs, forecasts = [], [], []
            with self.lock:
                # random.shuffle(self.dicts)
                self.init_count = 0

def train_epoch(lstm: nn.Module,
                optimizer: torch.optim.Optimizer,
                cuda_batches_queue: Queue,
                criterion: callable,
                dataset: MPWindDataset,
                epoch: int,
                steps_per_epoch: int,
                args):
    lstm.train()
    train_loss_avg = 0
    batch_averaged_loss = []
    num_batches = 0
    print(steps_per_epoch)
    for batch_idx in tqdm(range(steps_per_epoch), total=steps_per_epoch):

        gfs_seq, station_seq, target, true_forecast = cuda_batches_queue.get(block=True)
        y = lstm(gfs_seq, station_seq)
        loss = criterion(y, true_forecast)
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
    avg_loss = train_loss_avg / steps_per_epoch
    losses_dict = {'avg_loss': avg_loss}
    return losses_dict


def validate(lstm: nn.Module,
             cuda_batches_queue: Queue,
             criterion: callable,
             dataset: MPWindDataset,
             val_steps: int,
             args):
    lstm.eval()
    with torch.no_grad():
        val_loss_avg = 0
        for batch_idx in tqdm(range(val_steps), total=val_steps):
            gfs_seq, station_seq, target, true_forecast = cuda_batches_queue.get(block=True)
            y = lstm(gfs_seq, station_seq)
            # target = dataset.targetss.channel_inverse_transform(target, 2)
            loss = criterion(y, true_forecast)
            val_loss_avg += loss.item()
        avg_loss = val_loss_avg / val_steps
        losses_dict = {'avg_loss': avg_loss}
    return losses_dict


def main(args):
    # cuda = True if torch.cuda.is_available() else False
    # if cuda:
    #     torch.cuda.set_device(0)
    # cuda_dev = torch.device('cuda:0')
    # for GFS data cropped around Falconara station
    cfg = Config.fromfile(args.config_file)
    print(cfg)
    station_augmentation = [
        transforms.RandomApply([mytransforms.StationNormalNoize(0, 0.07), ], p=0.3),
    ]
    gfs_augmentation = torch.nn.Sequential(
        mytransforms.GFSRandomPosterize(6, 0.05),
        transforms.RandomApply(torch.nn.ModuleList([mytransforms.GFSNormalNoize(mean=0, std=0.04)]), p=0.2),
        mytransforms.GFSAffine(degrees=3, p=0.1),
        mytransforms.ChannelRandomErasing(p=0.07)
    )

    station_augmentation = transforms.Compose(station_augmentation)
    gfs_augmentation = transforms.Compose(gfs_augmentation)

    dataset = MPWindDataset(args.gfs_field_files, args.gfs_forecast_files, args.target_dir, args.sequence_length,
                            args.forecast_range, args.batch_size, station_augmentation, gfs_augmentation,
                            test_mode=False)
    val_ds = MPWindDataset(args.val_gfs_files, args.val_gfs_forecast_files, args.target_dir, args.sequence_length,
                           args.forecast_range, args.batch_size,
                           test_mode=True)
    steps_per_epoch = len(dataset) // args.batch_size + 1
    val_steps = len(val_ds) // args.batch_size + 1
    batches_queue_length = min(steps_per_epoch, 16)

    print('creating the model')

    model = build_model(args.model_type, cfg.model)
    model.to(device)
    print('lstm created')
    # if args.start_epoch != 1:
    #     lstm.load_state_dict(torch.load('epochs/netG_epoch_%d.pth' % (args.start_epoch - 1)))
    #     lstm.load_state_dict(torch.load('epochs/netD_epoch_%d.pth' % (args.start_epoch - 1)))
    # else:
    #     pass

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=10e-7,
                                                                     last_epoch=-1)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_gamma)

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

    val_batches_queue = Queue(maxsize=batches_queue_length)
    val_cuda_batches_queue = Queue(maxsize=4)
    val_thread_killer = thread_killer()
    val_thread_killer.set_tokill(False)

    for _ in range(args.num_workers):
        thr = Thread(target=threaded_batches_feeder, args=(val_thread_killer, val_batches_queue, val_ds))
        thr.start()
    val_cuda_transfers_thread_killer = thread_killer()
    val_cuda_transfers_thread_killer.set_tokill(False)
    val_cudathread = Thread(target=threaded_cuda_batches, args=(
        val_cuda_transfers_thread_killer, val_cuda_batches_queue, val_batches_queue, args.sequence_length))
    val_cudathread.start()

    print('\n\nstart training')
    # start_epoch = 1 if resume_state is None else resume_state.epoch
    avg_train_loss = []
    avg_val_loss = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        # print('\n\n%s: Train epoch: %d of %d' % (args.run_name, epoch, EPOCHS))

        print('Train epoch: %d of %d' % (epoch, args.epochs))
        train_losses = train_epoch(model, optimizer, train_cuda_batches_queue, criterion, dataset, epoch,
                                   steps_per_epoch, args)
        val_losses = validate(model, val_cuda_batches_queue, criterion, val_ds,
                              val_steps, args)
        print(f'epoch {epoch} train avg_loss', train_losses['avg_loss'])
        print(f'epoch {epoch} val avg_loss', val_losses['avg_loss'])

        avg_train_loss.append(train_losses['avg_loss'])
        avg_val_loss.append(val_losses['avg_loss'])
        with open('transformer/train.txt', 'a') as f:
            f.write(str(epoch) + ' ' + str(avg_train_loss[-1]) + '\n')
        with open('transformer/val.txt', 'a') as f:
            f.write(str(epoch) + ' ' + str(avg_val_loss[-1]) + '\n')
        torch.save(model.state_dict(), 'epochs_transformer/transformer_epoch_%d.pth' % epoch)
        if avg_val_loss[-1] == min(avg_val_loss):
            torch.save(model.state_dict(), 'epochs_transformer/transformer_best_epoch_%d.pth' % epoch)
        np.save('transformer/train_loss', np.array(avg_train_loss))
        np.save('transformer/val_loss', np.array(avg_val_loss))
        np.save('transformer/lr', scheduler.get_last_lr())
        print(scheduler.get_last_lr(), '- current learning rate')
        scheduler.step()

    train_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    val_thread_killer.set_tokill(True)
    val_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(args.num_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
            val_batches_queue.get(block=True, timeout=1)
            val_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass


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
    parser.add_argument('--save_name', type=str, default='transformer')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_false')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epochs', type=int, default=149)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='total number of batch size of labeled data')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--exp_gamma', type=float, default=0.955)

    '''
    Backbone Net Configurations
    '''

    # parser.add_argument('--fea_model_type', type=str, default='ResNet-18')
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--forecast_range', type=int, default=6)

    '''
    Data Configurations
    '''
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=509)
    parser.add_argument('--station_params_number', type=int, default=3)
    parser.add_argument('--sequence_length', type=int, default=72)
    parser.add_argument('--gfs_field_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_train.npy" for i in
                                 range(10)])
    parser.add_argument('--gfs_forecast_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_f006_train.npy" for i in
                                 range(10)])
    parser.add_argument('--target_dir', type=str,
                        default='/app/Windy/Station/')
    parser.add_argument('--val_gfs_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_val.npy" for i in
                                 range(18)])
    parser.add_argument('--val_gfs_forecast_files', type=list,
                        default=[f"/app/Windy/Station/GFS_falconara_param_{i}of10_15011500-22110212_f006_val.npy" for i
                                 in
                                 range(10)])
    parser.add_argument('--val_target_dir', type=str,
                        default='/app/Windy/Station/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    main(args)
