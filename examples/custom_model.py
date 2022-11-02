import torch
from nnprofiler import LayerProf, get_children


class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 1)
        self.linear2 = torch.nn.Linear(10, 10000)

    def forward(self, x):
        return self.linear2(x) + self.linear1(x)


net = MyNet()
input = torch.randn(16, 10)
# Warm-up
net(input).sum().backward()

with LayerProf(net) as prof:
    net(input).sum().backward()

    print(prof.layerwise_summary())
