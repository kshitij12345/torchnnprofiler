import torch
from nnprofiler import LayerProf, get_children


class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear2(x) + self.linear1(x)


net = MyNet()
prof = LayerProf(net)

for name, layer in get_children(net):
    prof.attach_backward_hook(name)

y = net(torch.randn(16, 10, requires_grad=True))
y.sum().backward()

y = net(torch.randn(16, 10, requires_grad=True))
y.sum().backward()
print(prof.get_timings())
