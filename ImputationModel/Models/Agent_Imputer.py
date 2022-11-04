import torch
from torch import nn
import ImputationModel.Models.Time_LSTM_Module as TimeLSTM
import ImputationModel.Models.GNN_Module as GCN

class seq_lstm(nn.Module):
    def __init__(self, input_size= 24, hidden_layer_size=100, output_size=50, batch_size=128):
        super().__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.lstm = TimeLSTM.TimeLSTM(input_size, hidden_layer_size, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.batch_size = batch_size
        self.relu = nn.ReLU()

    def forward(self, input_seq, ts):
        lstm_out = self.lstm(input_seq,ts)
        outs = self.linear(lstm_out[:,-1,:])
        outs = self.relu(outs)
        return outs

class AgentImputer(nn.Module):
    def __init__(self, input_size= 24, hidden_layer_size=100, output_size=2, batch_size=128):
        
        super().__init__()  
        self.hidden_layer_size = hidden_layer_size
        self.lstms = seq_lstm(input_size, hidden_layer_size)
        self.gcn = GCN.GCN(50)
        self.batch_size = batch_size
        self.relu = nn.ReLU()

    def forward(self, input_list, ts_list, edge_index):
        outputs = torch.cat([self.lstms(x,ts_l) for x, ts_l in zip(input_list, ts_list)],dim=1)
        outputs = outputs.reshape(outputs.shape[0],22,50)
        gcn_outputs = self.gcn(outputs, edge_index)
        return gcn_outputs