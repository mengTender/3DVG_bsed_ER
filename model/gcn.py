import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
import dgl
import dgl.nn as dglnn


class GCN(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim1=512, hidden_dim2=512, hidden_dim3=512, output_dim=512):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim1)
        self.conv2 = dglnn.GraphConv(hidden_dim1, hidden_dim2)
        self.conv3 = dglnn.GraphConv(hidden_dim2, hidden_dim3)
        self.conv4 = dglnn.GraphConv(hidden_dim3, output_dim)
        self.avg_pooling = dglnn.AvgPooling()

    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = self.avg_pooling(g, h)
        return h
