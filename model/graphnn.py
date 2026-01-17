

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

print(f'Torch version: {torch.__version__}')

class TrafficGNN(torch.nn.Module):
    """
    Pytorch Graph Neural Network architecture

    Input: Graph Data
    Output: Q-values for each avaliable road-light

    GATConv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATConv.html
    """
    def __init__(self, num_features, hidden_dim=64, output_dim=1):

        super(TrafficGNN, self).__init__()
        
        #Aggregates its own features and the features of the direct neighbours
        self.conv1 = GATConv(num_features, hidden_dim, heads=2, concat=False)

        #Aggreagates its own features and the features of neighbours up to 2 steps away. Can include more steps if neccesary in the future but for now simple is definitely better
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)

        #Take the processed graph information and output a single urgency score per road light
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        #Standard Forward pass with Activation function & Dropout included.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        scores = self.fc(x)

        return scores
    
#AI generated test case
if __name__ == "__main__":
    # Create dummy data: 4 Lanes (Nodes), 3 Features each
    dummy_x = torch.rand(4, 3) 
    # Create dummy edges: 0->1, 1->2, 2->3 (Linear chain)
    dummy_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    model = TrafficGNN(num_features=3)
    output = model(dummy_x, dummy_edges)
    
    print("Input Shape:", dummy_x.shape)
    print("Output Shape:", output.shape) # Should be [4, 1]
    print("Scores:", output.detach().numpy())