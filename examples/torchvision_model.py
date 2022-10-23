import torchvision
import torch
from nnprofiler import LayerProf, get_children

resnet = torchvision.models.resnet18(weights=None)

# Warm-up
inp = torch.randn(10, 3, 224, 224)
out = resnet(inp)

with LayerProf(resnet, profile_all_layers=False) as layer_prof:
    for name, layer in get_children(resnet):
        # Hack around
        # https://github.com/pytorch/pytorch/issues/61519
        if "relu" in name or "bn" in name:
            continue
        layer_prof.attach_backward_hook(name)

    resnet(inp).sum().backward()

    print(layer_prof.layerwise_summary())
