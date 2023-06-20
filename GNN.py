import numpy as np
from torch_geometric.nn import GCN, GAT, GraphSAGE
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.utils import train_test_split_edges
import torch
import argparse


class GAE(torch.nn.Module):
    def __init__(self, gnn_version):
        super(GAE, self).__init__()

        self.gnn_version = gnn_version

        self.linear1 = torch.nn.Linear(data.num_features, 32)  # First linear layer
        self.linear2 = torch.nn.Linear(32, 64)  # Second linear layer
        self.linear3 = torch.nn.Linear(64, 256)  # Second linear layer

        if gnn_version == 'gcn':
            self.conv = GCN(in_channels=256, hidden_channels=64,out_channels=64, num_layers=2)
        elif gnn_version == 'sage':
            self.conv = GraphSAGE(in_channels=256, hidden_channels=64,out_channels=64, num_layers=2)
        elif gnn_version == 'gat':
            self.conv = GAT(in_channels=256, hidden_channels=64,out_channels=64, num_layers=2)
        else:
            raise ValueError("Wrong GNN type")
        self.linear4 = torch.nn.Linear(64, 64)  # Third linear layer
        self.tanh = torch.nn.Tanh()  # Activation function

    def encode(self):
        x = self.linear1(data.x.float())  # First linear layer
        x = self.tanh(x)  # Activation function
        x = self.linear2(x)  # Second linear layer
        x = self.tanh(x)  # Activation function
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.conv(x, data.train_pos_edge_index)  # convolution 1
        x = self.linear4(x)

        return x

    def decode(self, z, pos_edge_index, neg_edge_index):  # Get the positive and negative edges
        # z contains the embeddings for all the nodes. Shape: (1842,2)
        # pos_edge_index = data.train_pos_edge_index. It contains ALL the training positive edges. shape: (2,8760)
        # neg_edge_index = It contains a sample of negative edges. SAME shape
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # concatenate positive and negative edges
        # edge_index, shape: (2,17520). It contains the positive (true) train edges and the negative (sampled) ones. START to END
        # edge_index[0] -> starting nodes of the links (mixed positive and negative edges)
        # edge_index[1] -> ending nodes of the links (mixed positive and negative edges)

        # z[edge_index[0]] retrieves the embeddings of the starting nodes for all edges
        # z[edge_index[1]] retrieves the embeddings of the ending nodes for all edges.

        # compute inner product between staring and ending node of each edge (all true positive and sampled negative edges)

        # The resulting logits represent the unnormalized scores for each edge.
        # Higher logits indicate a higher likelihood of an edge existing between the corresponding n
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()  # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list


def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equal to the length of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # positive edges
        num_nodes=data.num_nodes,  # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1))  # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()

    z = model.encode()  # encode
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # decode

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)  # ground truth labels
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def evaluate(mode=None):
    model.eval()

    if mode == 'test':
        pos_edge_index = data[f'test_pos_edge_index']
        neg_edge_index = data[f'test_neg_edge_index']
    elif mode == 'val':
        pos_edge_index = data[f'val_pos_edge_index']
        neg_edge_index = data[f'val_neg_edge_index']
    elif mode == 'train':
        pos_edge_index = data[f'train_pos_edge_index']
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,  # positive edges
            num_nodes=data.num_nodes,  # number of nodes
            num_neg_samples=data.train_pos_edge_index.size(1))  # number of neg_sample equal to number of pos_edges

    z = model.encode()
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    link_predictions = np.round(link_probs.cpu().detach().numpy()).astype(int)

    link_labels = get_link_labels(pos_edge_index,
                                  neg_edge_index)  # get link ground truths (Consider true and neg sampled edges)

    precision = precision_score(link_labels.cpu(), link_predictions)
    recall = recall_score(link_labels.cpu(), link_predictions)
    f1 = f1_score(link_labels.cpu(), link_predictions)
    roc_auc = roc_auc_score(link_labels.cpu(), link_predictions)
    return precision, recall, f1, roc_auc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='Specify which graph version to use')
    parser.add_argument('--gnn', type=str, help='Specify which gnn version to use') #gcn, sage, gat

    args = parser.parse_args()

    graph_version = args.graph
    gnn_version = args.gnn

    if graph_version == "medium":
        loaded_data = torch.load('MEDIUMgraph.pt')
    elif graph_version == "enhanced":
        loaded_data = torch.load('ENHANCEDgraph.pt')
    else:
        raise ValueError

    print(f"Number of node features: {loaded_data.x.size(1)}")

    data = train_test_split_edges(loaded_data)

    print(f"Training set size : {data.train_pos_edge_index.size(1)}")
    print(f"Validation set size : {data.val_pos_edge_index.size(1)}")
    print(f"Test set size : {data.test_pos_edge_index.size(1)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model, data = GAE(gnn_version).to(device), data.to(device)

    print(
        f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters\n")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    best_f1_perf = 0
    for epoch in range(1, 50):
        train_loss = train()
        precision, recall, f1, roc_auc = evaluate(mode="val")
        if f1 > best_f1_perf:
            best_val_perf = f1
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val f1: {:.4f}'
        if epoch % 5 == 0:
            print(log.format(epoch, train_loss, best_val_perf))

    # precision, recall, f1, roc_auc = evaluate(mode="train")
    # print("Final Train performance \n")
    # print(f"Precision : {precision}")
    # print(f"Recall : {recall}")
    # print(f"F1 score : {f1}")
    # print(f"ROC AUC Score : {roc_auc}")

    precision, recall, f1, roc_auc = evaluate(mode="test")
    print("Final Test performance \n")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    print(f"F1 score : {f1}")
    print(f"ROC AUC Score : {roc_auc}")
    # z = model.encode()
    # final_edge_index = model.decode_all(z)
