import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = "cpu"
class UnifiedLSTM(nn.Module):
    """Same as SepLSTM_2dir but sin and cos are restricted manually to satisfy basic trigonometric identity"""

    def __init__(self, gfs_input_size, station_input_size, station_output_size, gfs_hidden_size, num_layers, in_channels, forecast_range,
                 station_scaler):
        super(UnifiedLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.gfs_input_size = gfs_input_size  # input size
        self.hidden_size = gfs_hidden_size  # hidden state
        self.station_size = station_input_size
        self.station_output_size = station_output_size
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
        new_layer = nn.Conv2d(in_channels=in_channels,
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
        self.fc_1 = nn.Linear(256, self.forecast_range * self.station_output_size)  # fully connected last layer

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
        out = out.view(-1, self.forecast_range, self.station_output_size)
        print(out.shape)
        out = self.station_scaler.channel_inverse_transform(out)
        speed, direction = torch.split(out, [1, 2], dim=2)
        l2norm = torch.sqrt(torch.sum(torch.square(direction), 2)).unsqueeze(2)
        direction = direction / l2norm
        out = torch.cat((speed, direction), 2)

        return out

class WeatherLSTM(nn.Module):
    """Same as SepLSTM_2dir but sin and cos are restricted manually to satisfy basic trigonometric identity"""
    def __init__(self, build_fea_model, station_input_size, station_output_size, num_layers, forecast_range):
        super(WeatherLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.station_input_size = station_input_size
        self.station_output_size = station_output_size
        self.forecast_range = forecast_range

        self.fea_model, self.gfs_size = build_fea_model()

        self.gfsLstm = nn.LSTM(input_size=self.gfs_size, hidden_size=self.gfs_size,
                               num_layers=num_layers, batch_first=True)  # lstm
        self.stationLstm = nn.LSTM(input_size=station_input_size, hidden_size=station_input_size,
                                   num_layers=num_layers, batch_first=True)

        self.fc_3 = nn.Linear(self.gfs_size + self.station_input_size + self.forecast_range * self.gfs_size,
                              4096)  # fully connected 1
        self.bn3 = nn.BatchNorm1d(4096)
        self.softplus1 = nn.Softplus()
        self.dropout = nn.Dropout(p=0.1)
        self.fc_2 = nn.Linear(4096, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.softplus2 = nn.Softplus()
        self.fc_1 = nn.Linear(256, self.forecast_range * self.station_output_size)  # fully connected last layer

    def forward(self, gfs, station):
        gfs_batch_size = gfs.shape[0]
        gfs_seq_len = gfs.shape[1]
        gfs = gfs.view(-1, gfs.shape[-3], gfs.shape[-2], gfs.shape[-1])
        gfs = self.fea_model(gfs)
        gfs = gfs.view(gfs_batch_size, gfs_seq_len, -1)
        gfs, model_predict = torch.split(gfs, [72, self.forecast_range], dim=1)

        gfs_h_0 = Variable(torch.zeros(self.num_layers, gfs.size(0), self.gfs_size, device=device))
        gfs_c_0 = Variable(torch.zeros(self.num_layers, gfs.size(0), self.gfs_size, device=device))

        station_h_0 = Variable(
            torch.zeros(self.num_layers, station.size(0), self.station_input_size, device=device))  # hidden state
        station_c_0 = Variable(
            torch.zeros(self.num_layers, station.size(0), self.station_input_size, device=device))  # internal state
        # Propagate input through LSTM
        gfs_output, (gfs_hn, gfs_cn) = self.gfsLstm(gfs.float(), (
            gfs_h_0.float(), gfs_c_0.float()))  # lstm with input, hidden, and internal state
        gfs_hn = gfs_hn.view(-1, self.gfs_size)  # reshaping the data for Dense layer next batch_size*2048

        station_output, (station_hn, station_cn) = self.stationLstm(station.float(), (
            station_h_0.float(), station_c_0.float()))  # lstm with input, hidden, and internal state
        station_hn = station_hn.view(-1, self.station_input_size)  # reshaping the data for Dense layer next batch_size*2048
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
        out = out.view(-1, self.forecast_range, self.station_output_size)
        # out = self.station_scaler.channel_inverse_transform(out)
        speed, direction = torch.split(out, [1, 2], dim=2)
        l2norm = torch.sqrt(torch.sum(torch.square(direction), 2)).unsqueeze(2)
        direction = direction / l2norm
        out = torch.cat((speed, direction), 2)

        return out


class build_ResNet:
    def __init__(self, fea_model_cfg):
        self.in_channels = fea_model_cfg.in_channels
        self.out_channels = 512
        # self.out_config = {'out_channels': self.out_channels}

    def build(self):
        fea_model = models.resnet18()
        # Here we rebuild ResNet feature model to input 12 channels
        layer = fea_model.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)
        fea_model.conv1 = new_layer
        fea_model = torch.nn.Sequential(*(list(fea_model.children())[:-1])).to(device)
        return fea_model, self.out_channels


class MyTransformer(nn.Transformer):
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class MyTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, kdim, vdim, dropout=0.1, batch_first=True, device=None, dtype=None):
        super(MyTransformerDecoderLayer, self).__init__(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                        device=device, dtype=dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, kdim=kdim, vdim=vdim, dropout=dropout,
                                                    batch_first=batch_first,
                                                    **factory_kwargs)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=72):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class WindTransformer(nn.Module):
    def __init__(self, in_channels, dropout=0.1, decoder=None):
        super(WindTransformer, self).__init__()
        self.fea_model = models.resnet18()
        # Here we rebuild ResNet feature model to input 12 channels

        layer = self.fea_model.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=in_channels,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)

        self.fea_model.conv1 = new_layer
        # self.pos_encoder = PositionalEncoding(512, dropout)
        self.fea_model = torch.nn.Sequential(*(list(self.fea_model.children())[:-1])).to(device)
        decoder_layer = MyTransformerDecoderLayer(d_model=512, nhead=16, kdim=517, vdim=517)
        self.transformer = MyTransformer(d_model=517, nhead=11, num_encoder_layers=12, batch_first=True,
                                         custom_decoder=decoder_layer)
        self.fc = nn.Linear(512, 3)

    def forward(self, gfs, station):
        gfs_batch_size = gfs.shape[0]
        gfs_seq_len = gfs.shape[1]
        gfs = gfs.view(-1, gfs.shape[-3], gfs.shape[-2], gfs.shape[-1])
        gfs = self.fea_model(gfs)
        gfs = gfs.view(gfs_batch_size, gfs_seq_len, -1)
        # gfs = self.pos_encoder(gfs)
        gfs, model_predict = torch.split(gfs, [72, 6], dim=1)
        inputs = torch.cat((gfs, station), dim=2)
        out = self.transformer(inputs, model_predict)
        out = self.fc(out)
        speed, direction = torch.split(out, [1, 2], dim=2)
        l2norm = torch.sqrt(torch.sum(torch.square(direction), 2)).unsqueeze(2)
        direction = direction / l2norm
        out = torch.cat((speed, direction), 2)
        return out


class AttentionLSTM(nn.Module):
    def __init__(self):
        super(AttentionLSTM, self).__init__()
