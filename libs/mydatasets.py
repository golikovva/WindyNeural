from torch.utils.data import Dataset


class WindDataset(Dataset):
    def __init__(self, data, target, sequence_length, forecast_range, device, transform=None, target_transform=None):
        self.sequence_length = sequence_length
        self.forecast_range = forecast_range
        self.data = data
        self.target = target
        self.device = device

    def __len__(self):
        return self.data.shape[0]-self.sequence_length-self.forecast_range

    def __getitem__(self, idx):
        gfs_seq = self.data[idx:idx+self.sequence_length+self.forecast_range, :]
        train_station_seq = self.target[idx:idx+self.sequence_length]
        y_fcr = self.target[idx+self.sequence_length:idx+self.sequence_length+self.forecast_range]
        return gfs_seq.to(self.device), train_station_seq.to(self.device), y_fcr.to(self.device)