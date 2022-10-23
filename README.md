### nnprofiler

`nnprofiler` provides a utility class `LayerProf` to measure the forward and backward execution time of PyTorch's `nn.Module` which could be a single layer or a complete model from `transformers` ,`torchvision`, etc or your own custom model. It captures and provides the timings for all the layers present in the model.

#### Example
```python
import torch
from nnprofiler import LayerProf


class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 1)
        self.linear2 = torch.nn.Linear(10, 10000)

    def forward(self, x):
        return self.linear2(x) + self.linear1(x)

net = MyNet()
inp = torch.randn(16, 10)
# Warm-up
net(inp).sum().backward()

# This could be any model (torchvision, transformers or your custom model).
with LayerProf(net) as prof:
    net(inp).sum().backward()

    print(prof.layerwise_summary())
```

Output
```
MyNet(
  (linear1): Linear(Forward Time: 0.027650ms | Backward Time: 0.053989ms)
  (linear2): Linear(Forward Time: 0.195528ms | Backward Time: 0.524774ms)
)
```

As expected, we see that `linear2` takes much longer than `linear1` for both forward and backward. For more examples, checkout the `examples` directory.

**Note**: This is not a benchmarking utility like `timeit` or `pytorch.utils.benchmark` which run a piece of code multiple times to capture more accurate timings

#### Motivation

While training a model, it is important to know about the performance characteristics of the model, especially, if it will be deployed in production. To that end, knowing how long each layer takes for computation can help you find bottlenecks.

#### Why `nnprofiler` instead of `torch.profiler`?

PyTorch already ships with a utility to profile your code, so why another? It's simple that each of them target different use-case and work at different levels.

[torch.profiler](https://pytorch.org/docs/stable/profiler.html) helps profile the model at the granularity of PyTorch operators. This means that one has to guess which layer that belongs to and also if you use `conv` at multiple places then how long is each one taking.

Example of `torch.profiler`'s profile for `resnet18` from `torchvision`:
```
# ---------------------------------  ------------  ------------  ------------  ------------
#                              Name      Self CPU     CPU total  CPU time avg    # of Calls
# ---------------------------------  ------------  ------------  ------------  ------------
#                   model_inference       5.509ms      57.503ms      57.503ms             1
#                      aten::conv2d     231.000us      31.931ms       1.597ms            20
#                 aten::convolution     250.000us      31.700ms       1.585ms            20
#                aten::_convolution     336.000us      31.450ms       1.573ms            20
#          aten::mkldnn_convolution      30.838ms      31.114ms       1.556ms            20
#                  aten::batch_norm     211.000us      14.693ms     734.650us            20
#      aten::_batch_norm_impl_index     319.000us      14.482ms     724.100us            20
#           aten::native_batch_norm       9.229ms      14.109ms     705.450us            20
#                        aten::mean     332.000us       2.631ms     125.286us            21
#                      aten::select       1.668ms       2.292ms       8.988us           255
# ---------------------------------  ------------  ------------  ------------  ------------
# Self CPU time total: 57.549ms
```

Running `LayerProf` on `resnet18` as
```python
import torchvision
import torch
from nnprofiler import LayerProf, get_children

resnet = torchvision.models.resnet18(weights=None)

# Warm-up
inp = torch.randn(10, 3, 224, 224)
out = resnet(inp)

with LayerProf(resnet, profile_all_layers=False) as layer_prof:
    names_and_layers = list(get_children(resnet))
    for idx, (name, layer) in enumerate(names_and_layers):
        # Hack around 
        # https://github.com/pytorch/pytorch/issues/61519
        if "relu" in name or "bn" in name:
            continue
        layer_prof.attach_backward_hook(name)

    out = resnet(inp)
    out.sum().backward()

    print(layer_prof.layerwise_summary())

```

Output
```
ResNet(
  (conv1): Conv2d(Forward Time: 121.985564ms | Backward Time: 0.012790ms)
  (bn1): BatchNorm2d()
  (relu): ReLU()
  (maxpool): MaxPool2d(Forward Time: 125.893256ms | Backward Time: 125.614427ms)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(Forward Time: 1.803901ms | Backward Time: 127.819536ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 126.037036ms | Backward Time: 127.432800ms)
      (bn2): BatchNorm2d()
    )
    (1): BasicBlock(
      (conv1): Conv2d(Forward Time: 1.873590ms | Backward Time: 127.779866ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 1.842620ms | Backward Time: 128.236670ms)
      (bn2): BatchNorm2d()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(Forward Time: 125.681849ms | Backward Time: 3.020898ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 1.290446ms | Backward Time: 127.308880ms)
      (bn2): BatchNorm2d()
      (downsample): Sequential(
        (0): Conv2d(Forward Time: 0.603593ms | Backward Time: 125.808026ms)
        (1): BatchNorm2d(Forward Time: 0.231657ms | Backward Time: 0.262567ms)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(Forward Time: 125.161703ms | Backward Time: 127.133824ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 1.202197ms | Backward Time: 127.814146ms)
      (bn2): BatchNorm2d()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(Forward Time: 0.995109ms | Backward Time: 122.874559ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 1.301246ms | Backward Time: 2.906958ms)
      (bn2): BatchNorm2d()
      (downsample): Sequential(
        (0): Conv2d(Forward Time: 0.457235ms | Backward Time: 125.288073ms)
        (1): BatchNorm2d(Forward Time: 0.153528ms | Backward Time: 0.160939ms)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(Forward Time: 1.265666ms | Backward Time: 126.964594ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 125.179764ms | Backward Time: 3.120596ms)
      (bn2): BatchNorm2d()
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(Forward Time: 1.162807ms | Backward Time: 126.219883ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 126.034236ms | Backward Time: 62.486642ms)
      (bn2): BatchNorm2d()
      (downsample): Sequential(
        (0): Conv2d(Forward Time: 0.462975ms | Backward Time: 63.839558ms)
        (1): BatchNorm2d(Forward Time: 0.122288ms | Backward Time: 0.097689ms)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(Forward Time: 125.842666ms | Backward Time: 126.943985ms)
      (bn1): BatchNorm2d()
      (relu): ReLU()
      (conv2): Conv2d(Forward Time: 1.722881ms | Backward Time: 74.761527ms)
      (bn2): BatchNorm2d()
    )
  )
  (avgpool): AdaptiveAvgPool2d(Forward Time: 0.121119ms | Backward Time: 0.083329ms)
  (fc): Linear(Forward Time: 0.181158ms | Backward Time: 9.337799ms)
)
```

NOTE: That we are unable to capture the timings for `bn` and `RELU` because of https://github.com/pytorch/pytorch/issues/61519

#### IMPORTANT: The hooks mechanism that we utilize for timing the backward pass is only available on the nightly version of PyTorch and will take a few months to be released in the stable version.
