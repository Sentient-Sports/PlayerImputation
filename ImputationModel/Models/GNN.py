import ImputationModel.Models.GNN_Module as GCN
import torch
from torch import nn

class GNN(nn.Module):
    def __init__(self, input_size= 25, hidden_layer_size=100, output_size=2, batch_size=128):
        
        super().__init__()  
        self.hidden_layer_size = hidden_layer_size
        self.gcn = GCN.GCN(2)
        self.batch_size = batch_size
        self.relu = nn.ReLU()

    def forward(self, input_list, ts_list, edge_index):
        outputs = torch.cat([torch.tensor(x[:,-3,:]).reshape(x.shape[0],1,25) for x in input_list],dim=1)
        gcn_outputs = self.gcn(outputs, edge_index)
        return gcn_outputs