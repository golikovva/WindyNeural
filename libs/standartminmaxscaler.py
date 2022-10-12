import torch


class MyStandartScaler:
    def __init__(self):
        self.channel_means = None
        self.channel_stddevs = None
        self.channels = None
        self.mean = None
        self.stddev = None
        self.channels_dim = None

    def fit_transform(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def channel_fit_transform(self, tensor, channels=None, channels_dim=1):
        self.channels_dim = channels_dim
        self.channel_means = []
        self.channel_stddevs = []
        if not channels:
            self.channels = range(tensor.shape[1])
        else:
            self.channels = channels
        tensor = torch.split(tensor, 1, dim=channels_dim)
        for i, channel in enumerate(tensor):
            if i in self.channels:
                self.channel_means.append(torch.mean(channel))
                self.channel_stddevs.append(torch.std(channel))
                channel -= self.channel_means[-1]
                channel /= self.channel_stddevs[-1]
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def channel_inverse_transform(self, tensor, channels_dim=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        for i in range(len(self.channels)):
            tensor[self.channels[i]] = tensor[self.channels[i]] * self.channel_stddevs[i]
            tensor[self.channels[i]] = tensor[self.channels[i]] + self.channel_means[i]
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def channel_fit(self, tensor, channels=None, channels_dim=1):
        self.channels_dim = channels_dim
        self.channel_means = []
        self.channel_stddevs = []
        if not channels:
            self.channels = range(tensor.shape[1])
        else:
            self.channels = channels
        tensor = torch.split(tensor, 1, dim=self.channels_dim)
        for i, channel in enumerate(tensor):
            if i in self.channels:
                self.channel_means.append(torch.mean(channel))
                self.channel_stddevs.append(torch.std(channel))

    def channel_transform(self, tensor, channels_dim=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        tensor = torch.split(tensor, 1, dim=channels_dim)
        j = 0
        for i, channel in enumerate(tensor):
            if i in self.channels:
                channel -= self.channel_means[j]
                channel /= self.channel_stddevs[j]
                j += 1
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def fit(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        self.max = tensor.max()

    def transform(self, tensor):
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def flip_transform(self, tensor):
        tensor = -tensor
        tensor += self.max
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def inverse_transform(self, tensor):
        tensor *= self.stddev
        tensor += self.mean
        return tensor


class MyMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.range_min = feature_range[0]
        self.range_max = feature_range[1]
        self.tensor_min = None
        self.tensor_max = None
        self.channel_mins = []
        self.channel_maxs = []

    def channel_fit_transform(self, tensor):
        self.channel_mins = []
        self.channel_maxs = []
        # assumed data size: Bs x C x H x W, were Bs - batch size, C - channels, H,W are height and width
        for i in range(tensor.shape[1]):
            self.channel_mins.append(torch.min(tensor[:, i]))
            self.channel_maxs.append(torch.max(tensor[:, i]))
            c_std = (tensor[:, i] - self.channel_mins[-1]) / (self.channel_maxs[-1] - self.channel_mins[-1])
            tensor[:, i] = c_std
        return tensor

    def fit_transform(self, tensor):
        self.tensor_min = torch.min(tensor)
        self.tensor_max = torch.max(tensor)
        X_std = (tensor - self.tensor_min) / (self.tensor_max - self.tensor_min)
        X_scaled = X_std * (self.range_max - self.range_min) + self.range_min
        return X_scaled

    def inverse_transform(self, tensor):
        if self.channel_mins is not None:
            for i in range(len(tensor)):
                tensor[i] = tensor[i] * (self.channel_maxs[i] - self.channel_mins[i]) + self.channel_mins[i]
            return tensor
        return tensor * (self.tensor_max - self.tensor_min) + self.tensor_min

    def channel_batch_fit_transform(self, tensor, channels_i=1):
        self.channel_mins = []
        self.channel_maxs = []

        for i in range(tensor.shape[1]):
            self.channel_mins.append(torch.min(tensor[:, i]))
            self.channel_maxs.append(torch.max(tensor[:, i]))
            c_std = (tensor[:, i] - self.channel_mins[-1]) / (self.channel_maxs[-1] - self.channel_mins[-1])
            tensor[:, i] = c_std

    def batch_inverse_transform(self, tensor):
        if self.channel_mins is not None:
            for i in range(len(self.channel_mins)):
                tensor[:, i] = tensor[:, i] * (self.channel_maxs[i] - self.channel_mins[i]) + self.channel_mins[i]
            return tensor
        return tensor * (self.tensor_max - self.tensor_min) + self.tensor_min
