import torch
import torch.nn as nn
import torch.nn.functional as F


class SAConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAConv, self).__init__()

        # MLP for self-augment feature (to create Δfi)
        self.self_augment_mlp = nn.Sequential(
            nn.Linear(in_channels, 3)
        )

        # MLP for edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels // 2)
          
        )

        # MLP after aggregation
        self.last_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels)
        )

        # Projection for residual connection
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, features, X, knn_graph):
        B, N, k, C = knn_graph.size()
        self.features = features

        # Compute self-augment feature Δfi and expand it for each neighbor
        triangle_fi = self.self_augment_mlp(self.features)
        triangle_fi = triangle_fi.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, 3)

        # Compute edge features
        e_1 = X + triangle_fi
        e_2 = knn_graph  # (B, N, k, in_channels)
        edge_features = torch.cat([e_1, e_2], dim=-1)
        x = self.edge_mlp(edge_features)

        # Local mixed aggregation
        x_max = x.max(dim=2)[0]
        x_mean = x.mean(dim=2)
        mixed_aggregation = torch.cat([x_max, x_mean], dim=-1)
        final_output = self.last_mlp(mixed_aggregation)

        features = self.projection(features)
        # Concatenate with self.features
        final_output = final_output + features

        return final_output

class SACNN(nn.Module):
    def __init__(self, num_classes, k=25):
        super(SACNN, self).__init__()
        self.k = k

        # Use ModuleList instead of a regular list
        self.sa_block = nn.ModuleList([
            #Paper Model
            SAConv(3, 64),
            SAConv(64, 128),
            SAConv(128, 256),

            #For 1024
            # SAConv(3, 32),
            # SAConv(32, 64),
            # SAConv(64, 128)
        ])

        # MLP after concatenation
        self.mlp = nn.Sequential(
            nn.Linear(448, 1024),  # Input: Concatenated features (64 + 128 + 256 = 448) paper Implementation
            #nn.Linear(448, 512), # Test 2
            #Best for 1024
            # nn.Linear(224, 512), # Test 3
            # nn.ReLU()
        )
        
        # Classification part besser ohne Batchnorm
        self.classfier = nn.Sequential(
            # Classifier in Paper 
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            #Classifier worked for 1024
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        self.features = features
        # List to store intermediate self.features
        self.feature_list = []
        # Feature extraction within Multi-SAConv Block (Input dim = Batch_size x Num_points x 3)
        for layer in self.sa_block:
            knn_graph = self.get_knn_graph(self.features)

            #Caluclation of X = Relative coordinates xi - xi' for each neighbor (B, N, k, 3)
            if len(self.feature_list) == 0:
                X = self.features.unsqueeze(2).expand(-1, -1, self.k, -1) - knn_graph
            
            self.features = layer(self.features, X, knn_graph)
            self.feature_list.append(self.features)
        
        # Concatenate intermediate features (Residual Connection)
        x = torch.cat(self.feature_list, dim=-1)
        x = self.mlp(x)

        # Classification
        # Max and mean pooling
        x_max = x.max(dim=1)[0]
        x_mean = x.mean(dim=1)   

        # Concatenate pooled features
        x = torch.cat([x_max, x_mean], dim=1) 
        x = self.classfier(x)
        return x

    def get_knn_graph(self, x):
        # This computes the Euclidean distance matrix for each point in the batch.
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx + inner + xx.transpose(2, 1)

        # Find the indices of the k nearest neighbors for each point
        _, idx = pairwise_distance.topk(k=self.k, dim=-1, largest=False)  # (B, N, k)

        # Create idx_base tensor to handle indexing for the batch
        idx_base = torch.arange(0, x.size(0), device=x.device).view(-1, 1, 1) * x.size(1)
        idx = idx + idx_base
        idx = idx.view(-1)

        # Gather the k-nearest neighbors
        x_flat = x.view(-1, x.size(-1))
        knn_graph = x_flat[idx, :]
        knn_graph = knn_graph.view(x.size(0), x.size(1), self.k, -1)
        return knn_graph
