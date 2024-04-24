import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, lstm_output):
        u = torch.tanh(torch.matmul(lstm_output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = lstm_output * att_score
        context = torch.sum(scored_x, dim=1)
        return context

class model(nn.Module):
    def __init__(self,out_channels,kernel_size,stride,hidden_size):
        super(model, self).__init__()

        self.lstm_x1 = nn.LSTM(input_size=4, hidden_size=12,num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_x1_2 = nn.LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)

        self.conv1 = nn.Conv1d(in_channels=12*2, out_channels=24, kernel_size=kernel_size,padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=kernel_size, padding=1, stride=stride,bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=out_channels)
        self.conv_op = nn.Conv1d(in_channels=out_channels, out_channels=out_channels + 3, kernel_size=kernel_size,padding=1, stride=stride, bias=False)
        self.conv_op2 = nn.Conv1d(in_channels=out_channels + 3, out_channels=out_channels + 3, kernel_size=kernel_size,padding=1, stride=stride, bias=False)
        self.lstm = nn.LSTM(input_size=out_channels+3, hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)  # input_size=912
        self.lstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)

        self.embedding2 = nn.Embedding(num_embeddings=16, embedding_dim=24)
        self.convx3_1 = nn.Conv1d(in_channels=24, out_channels=12, kernel_size=3, stride=stride, bias=False)
        self.convx3_2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=stride, bias=False)
        self.lstm_x3_1 = nn.LSTM(input_size=24, hidden_size=12,num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_x3_2 = nn.LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)
        self.attention=Attention(hidden_size)

        self.fc1 = nn.Linear(3816,160)
        self.fc2 = nn.Linear(160, 1)
        self.dropout=nn.Dropout(0.5)
        self.ln = nn.LayerNorm(24)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.bn3 = nn.BatchNorm1d(out_channels+3)
        self.bn4 = nn.BatchNorm1d(12)
        self.bn5=nn.BatchNorm1d(24)

        self.sigmoid = nn.Sigmoid()

        self.conv1x1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels + 3, kernel_size=1, stride=stride)

    def forward(self, x1,x2,x3):


        x1, _ = self.lstm_x1(x1)
        x1= self.dropout(x1)
        x1, _ =self.lstm_x1_2(x1)

        x2 = self.embedding(x2.long())
        x2 = self.bn1(x2.permute(0,2,1))
        original = self.conv1x1(x2)

        x2 = self.relu(self.conv_op(x2))
        x2 = self.dropout(x2)
        x2 = self.dropout(self.bn3(original+x2))

        x2 = self.relu(self.conv_op2(x2))
        x2 = self.dropout(x2)
        x2, _ = self.lstm(x2.permute(0,2,1))
        x2 = self.bn2(x2.permute(0, 2, 1))
        x2, _ = self.lstm2(x2.permute(0, 2, 1))

        x3 = self.embedding2(x3.long())
        x3 = self.bn5(x3.permute(0,2,1))

        x3 = self.convx3_1(x3)
        x3 = self.dropout((self.relu(x3)))
        x3 = self.relu((self.convx3_2(x3)))
        x3 = self.dropout(x3)

        x = torch.cat([x1, x2, x3.permute(0,2,1)], axis=1)  # 传统特征和预训练特征 结合:([64, 30, 106]),([64, 1, 54])
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.sigmoid(x)
