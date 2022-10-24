import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class UnifiedLSTM(nn.Module):
    """Same as SepLSTM_2dir but sin and cos are restricted manually to satisfy basic trigonometric identity"""

    def __init__(self, gfs_input_size, station_input_size, gfs_hidden_size, num_layers, new_in_channels, forecast_range, station_scaler):
        super(UnifiedLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.gfs_input_size = gfs_input_size  # input size
        self.hidden_size = gfs_hidden_size  # hidden state
        self.station_size = station_input_size
        self.forecast_range = forecast_range
        self.station_scaler = station_scaler
        for channels in self.station_scaler.channel_means:
            channels.to(device)
        for channels in self.station_scaler.channel_stddevs:
            channels.to(device)
        # self.station_scaler.channel_means = self.station_scaler.channel_means.to(device)
        # self.station_scaler.stddev = self.station_scaler.stddev.to(device)

        self.fea_model = models.resnet18()
        # Here we rebuild ResNet feature model to input 12 channels

        layer = self.fea_model.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=new_in_channels,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)

        self.fea_model.conv1 = new_layer
        self.fea_model = torch.nn.Sequential(*(list(self.fea_model.children())[:-1])).to(device)

        self.gfsLstm = nn.LSTM(input_size=gfs_input_size, hidden_size=gfs_hidden_size,
                               num_layers=num_layers, batch_first=True)  # lstm
        self.stationLstm = nn.LSTM(input_size=station_input_size, hidden_size=station_input_size,
                                   num_layers=num_layers, batch_first=True)

        self.fc_3 = nn.Linear(gfs_hidden_size + station_input_size + forecast_range * gfs_input_size,
                              4096)  # fully connected 1
        self.bn3 = nn.BatchNorm1d(4096)
        self.softplus1 = nn.Softplus()
        self.dropout = nn.Dropout(p=0.1)
        self.fc_2 = nn.Linear(4096, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.softplus2 = nn.Softplus()
        self.fc_1 = nn.Linear(256, self.forecast_range * self.station_size)  # fully connected last layer

    def forward(self, gfs, station):
        gfs_batch_size = gfs.shape[0]
        gfs_seq_len = gfs.shape[1]
        gfs = gfs.view(-1, gfs.shape[-3], gfs.shape[-2], gfs.shape[-1])
        gfs = self.fea_model(gfs)
        gfs = gfs.view(gfs_batch_size, gfs_seq_len, -1)
        gfs, model_predict = torch.split(gfs, [72, self.forecast_range], dim=1)

        gfs_h_0 = Variable(torch.zeros(self.num_layers, gfs.size(0), self.hidden_size, device=device))  # hidden state
        gfs_c_0 = Variable(torch.zeros(self.num_layers, gfs.size(0), self.hidden_size, device=device))  # internal state

        station_h_0 = Variable(
            torch.zeros(self.num_layers, station.size(0), self.station_size, device=device))  # hidden state
        station_c_0 = Variable(
            torch.zeros(self.num_layers, station.size(0), self.station_size, device=device))  # internal state
        # Propagate input through LSTM
        gfs_output, (gfs_hn, gfs_cn) = self.gfsLstm(gfs.float(), (
        gfs_h_0.float(), gfs_c_0.float()))  # lstm with input, hidden, and internal state
        gfs_hn = gfs_hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next batch_size*2048

        station_output, (station_hn, station_cn) = self.stationLstm(station.float(), (
        station_h_0.float(), station_c_0.float()))  # lstm with input, hidden, and internal state
        station_hn = station_hn.view(-1, self.station_size)  # reshaping the data for Dense layer next batch_size*2048
        model_predict = model_predict.view((model_predict.shape[0], model_predict.shape[1] * model_predict.shape[2]))
        in_tensor = torch.cat((model_predict, gfs_hn, station_hn), 1)

        out = self.fc_3(in_tensor)  # first Dense
        out = self.bn3(out)
        out = self.softplus1(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        out = self.bn2(out)
        out = self.softplus2(out)
        out = self.fc_1(out)
        out = out.view(-1, self.forecast_range, self.station_size)
        print(out.shape, 'lstm stat shape')
        out = self.station_scaler.channel_inverse_transform(out)
        speed, direction = torch.split(out, [1, 2], dim=2)
        l2norm = torch.sqrt(torch.sum(torch.square(direction), 2)).unsqueeze(2)
        direction = direction / l2norm
        out = torch.cat((speed, direction), 2)

        return out
