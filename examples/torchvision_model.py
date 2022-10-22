import torchvision
import torch
from nnprofiler import LayerProf, get_children

resnet = torchvision.models.resnet50(weights=None)

# Warm-up
inp = torch.randn(10, 3, 224, 224)
out = resnet(inp)

with LayerProf(resnet) as layer_prof:
    for name, layer in get_children(resnet):
        if "relu" in name or "bn" in name:
            continue
        layer_prof.attach_backward_hook(name)

    out = resnet(inp)
    out.sum().backward()
    layer_prof.get_timings()
    print(resnet)
