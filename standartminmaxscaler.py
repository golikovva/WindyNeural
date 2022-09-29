import torch


class MyStandartScaler:
    def __init__(self):
        self.mean = None
        self.stddev = None

    def fit_transform(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        tensor -= self.mean
        tensor /= self.stddev
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
        self.channel_mins = None
        self.channel_maxs = None

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