import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ScConv import ScConv 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Modify this to use 1-dimensional adaptive average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # Modify this, there is no longer a fourth dimension
        y = self.avg_pool(x).view(b, c) # Batch 32
        # y = self.avg_pool(x)    # Batch 32 1
        y = self.fc(y).view(b, c, 1)  # Modify this and no longer expand the fourth dimension.
        return x * y.expand_as(x)
    
class ResidualConv1D(nn.Module):
    """
    ResidualConv1D for use with best performing classifier in PyTorch
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool=False, se = False):
        super(ResidualConv1D, self).__init__()
        self.pool = pool
        self.se = se

        if self.pool:
            self.pool_layer = nn.AvgPool1d(kernel_size=2, stride=2)  # creates an average pooling layer, the example parameters here are kernel_size=2, stride=2
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.res_pool   = nn.AvgPool1d(kernel_size=2, stride=2)

        if self.se:
            self.se_layer = SELayer(out_channels)


        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same')

    def forward(self, x):
        res = x     # res1 Batch 32 1024

        if self.pool:
            x = self.pool_layer(x)      # Batch 32 512
            res = self.res_conv(res)    # res1 Batch 64 1024
            res = self.res_pool(res)    # res1 Batch 64 512

        out = self.conv1(x)             # res1 Batch 64 512
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.se:
            out = self.se_layer(out)    # res1 Batch 64 512
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)           # res1 Batch 64 512

        out += res
        return out

class Conv1dReduce(nn.Module):
    def __init__(self, in_channels=1024, out_channels=32, kernel_size=7):
        super(Conv1dReduce, self).__init__()
        # uses 1D convolution to reduce the 1024 channels to 32, and the convolution kernel size is 1
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same')

    def forward(self, x):
        return self.conv1d(x)

class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AudioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        # LSTM layer
        output, (h,c) = self.lstm(x)    # output 16 1024 64
        
        # # Take the output of the last time step of LSTM
        # output = output[:, -1, :]   # 16 64

        return output


class DeepConvLSTMModel(nn.Module):
    def __init__(self, n_features, n_classes, regression):
        super(DeepConvLSTMModel, self).__init__()
        self.regression = regression
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=16, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.res1 = ResidualConv1D(32, 64, 65, pool=True, se=True)
        self.res2 = ResidualConv1D(64, 64, 65)


        self.res3 = ResidualConv1D(64, 128, 33, pool=True)
        self.res4 = ResidualConv1D(128, 128, 33)

        self.res5 = ResidualConv1D(128, 256, 15, pool=True)
        self.res6 = ResidualConv1D(256, 256, 15)


        self.res7 = ResidualConv1D(256, 512, 7, pool=True)
        self.res8 = ResidualConv1D(512, 512, 7)

        self.res9 = ResidualConv1D(512, 1024, 3, pool=True)
        self.res10 = ResidualConv1D(1024, 1024, 3)


        self.maxpool = nn.MaxPool1d(kernel_size=16, stride=16)
        self.lstm = AudioLSTM(input_size=32, hidden_size=32)
        self.ScConv = ScConv(1024)

        # -----------------------------------------Add gap configuration---------------------------------------------
        # # GAP layer added here
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.x1024_32 = nn.Linear(1024, 32)
        # self.x32_3 = nn.Linear(32, 3)
        # ---------------------------------------------------------------------------------------

        self.myconv1024_32 = Conv1dReduce(in_channels=1024, out_channels=32, kernel_size=7)
        self.myconv32_1 = Conv1dReduce(in_channels=32, out_channels=1, kernel_size=7)
        self.myconv64_32 = Conv1dReduce(in_channels=64, out_channels=32, kernel_size=7)
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(64, n_classes)


    def forward(self, x):
        # # Assume that the shape of x is (N, C, L)
        # N, C, L = x.shape

        # H, W = 16, 16 # Target size
        # if L >= H * W:
        #     x_reshaped = x[:, :, :H * W].view(N, C, H, W)
        # else:
        #     raise ValueError("Cannot reshape input tensor to the desired output shape.")

        # Take training sentences as an example. If each word is a 100-dimensional vector, each sentence contains 24 words, and 10 sentences are trained at a time. Then batch_size=10,seq=24,input_size=100
        # By analogy, it should be 16 1024 x. Each point has x dimensions, and a curve has 1024 points. Train 16 sentences at a time. Then batch_size=16,seq=1024,input_size=x
        # This will cause another problem. 1024 points are too many for a curve. LSTM may not be able to remember it and it will not affect it. Originally, 16 1024 256 was passed in.
        # At first x 16 1 1024 Something seems wrong
        x = x.repeat(1, 2, 1) # Batch 2 1024
        x = self.conv1(x)   # Batch 32 1024
        x = self.bn1(x)
        x = self.relu(x)
        
        # Apply residual layer
        x = self.res1(x)    # Batch 64 512
        x = self.res2(x)

        x = self.res3(x)    # Batch 128 256
        x = self.res4(x)

        x = self.res5(x)    # Batch 256 128
        x = self.res6(x)

        x = self.res7(x)    # Batch 512 64 
        x = self.res8(x) 
        x = self.res9(x)    # Batch 1024 32 
        x = self.res10(x) 
        # x = self.res11(x)    # Batch 1024 16 
        # x = self.res12(x) 
        # x1 = self.maxpool(x) # Batch 1024 1   
        # x = torch.transpose(x, 1, 2) # Batch 1024 256 (batch, seq, input_size)


        # Follow it as written
        # x = self.ScConv(x)                  #   Batch 1024 256
        x_lstm = self.lstm(x)               #   Batch 1024 64
        x = self.myconv1024_32(x_lstm)        #   Batch 32 64 
        x = self.myconv32_1(x)              #   Batch 1 64
        ######################################


        x = self.dropout(x)
        x = torch.flatten(x, 1) # Batch 64
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)     # Batch 2 

        return x_lstm, x


