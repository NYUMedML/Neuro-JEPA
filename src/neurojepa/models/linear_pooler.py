from torch import nn


class Linear_Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    
class MultiLinear(nn.Module):
    def __init__(self, embed_dim, num_classes, device):
        super().__init__()
        self.classifiers = nn.ModuleList([
            Linear_Classifier(
                dim=embed_dim,
                num_classes=1,
            ).to(device)
            for _ in range(num_classes)
        ])

    def forward(self, x):
        return [m(x) for m in self.classifiers]