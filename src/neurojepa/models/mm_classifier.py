from torch import nn


class MultiModalLateFusion(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        embed_dim = 256
        
        self.bn1 = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.bn2 = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        
        self.proj1 = ProjectionHead(
            embedding_dim=128,
            projection_dim=embed_dim,
            output_dim=embed_dim,
            dropout=0.1,
        )
        self.proj2 = ProjectionHead(
            embedding_dim=128,
            projection_dim=embed_dim,
            output_dim=embed_dim,
            dropout=0.1,
        )

        self.act = nn.GELU()
        
        self.fusion = nn.Linear(embed_dim, num_classes)

    def forward(self, x1, x2):
        x_1 = self.bn1(self.proj1(x1))
        x_2 = self.bn2(self.proj2(x2))

        x_3 = self.fusion(self.act(x_1 + x_2))

        return x_3
    

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x