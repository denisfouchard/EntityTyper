import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv


class GraphEncoder(torch.nn.Module):
    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        raise NotImplementedError


class GCNEncoder(GraphEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_heads: int = -1,
    ):
        super(GCNEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GraphTransformerEncoder(GraphEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_heads: int = -1,
    ):
        super(GraphTransformerEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    edge_dim=in_channels,
                    dropout=dropout,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


class GATEncoder(GraphEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 4,
    ):
        super(GATEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, concat=False)
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GATConv(hidden_channels, out_channels, heads=num_heads, concat=False)
        )
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


GNN_MODEL_MAPPING: dict[str, GraphEncoder] = {
    "gcn": GCNEncoder,
    "gat": GATEncoder,
    "gt": GraphTransformerEncoder,
}
