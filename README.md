### nnprofiler

`nnprofiler` provides a utility class `LayerProf` to measure the forward and backward execution time of PyTorch's `nn.Module` which could be a single layer or a complete model from `transformers` ,`torchvision`, etc or your own custom model. It captures and provides the timings for all the layers present in the model.

#### Example
```python
import torch
import torch.nn as nn
class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10,1))
        self.linear1 = torch.nn.Linear(10, 11)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = nn.ModuleList()
        self.linear3.append(torch.nn.Linear(11, 12))
        self.linear3.append(torch.nn.Linear(12, 11))
        self.linear4 = nn.Sequential(nn.Linear(11,7),nn.Tanh(),nn.Linear(7,8))
        self.linear5 = nn.ModuleDict()
        self.linear5['a'] = nn.Linear(8,3)
        self.linear5['b'] = nn.Linear(3,1)
    def forward(self, x):
        x= torch.nn.functional.tanh(self.linear2(x)@self.weight ) + self.linear3[0](self.linear1(x))
        x= self.linear3[1](x)
        x= self.linear4(x)
        x= self.linear5['a'](x)
        x= self.linear5['b'](x)
        return x

net = MyNet()
inp = torch.randn(16, 10)
# Warm-up
net(inp).sum().backward()

# This could be any model (torchvision, transformers or your custom model).
with LayerProf(net) as prof:
    net(inp).sum().backward()
    summary_str = prof.layerwise_summary()
    pool = prof.layerwise_timepool()

```
- A string summary in `summary_str = prof.layerwise_summary()` like
  ```
  print(summary_str)
  ---------------------------------------------------------
  MyNet【Forward Time: 2.098945ms|Backward Time: 0.010771ms】(
    (linear1): Linear(Forward Time: 0.041632ms|Backward Time: 0.014860ms)
    (linear2): Linear(Forward Time: 0.621580ms|Backward Time: 0.009877ms)
    (linear3): ModuleList(
      (0): Linear(Forward Time: 0.054222ms|Backward Time: 0.048355ms)
      (1): Linear(Forward Time: 0.053606ms|Backward Time: 0.050200ms)
    )
    (linear4): Sequential【Forward Time: 0.315535ms|Backward Time: 0.229794ms】(
      (0): Linear(Forward Time: 0.069707ms|Backward Time: 0.070743ms)
      (1): Tanh(Forward Time: 0.044920ms|Backward Time: 0.039867ms)
      (2): Linear(Forward Time: 0.048994ms|Backward Time: 0.046925ms)
    )
    (linear5): ModuleDict(
      (a): Linear(Forward Time: 0.045158ms|Backward Time: 0.084371ms)
      (b): Linear(Forward Time: 0.046064ms|Backward Time: 0.099857ms)
    )
  )
  ```
- A tree-like time count in `pool=prof.layerwise_timepool()`
  ```
  from nnprofiler.visulization import print_namespace_tree
  print_namespace_tree(pool)
  -----------------------------------------------------------
  MyNet
    forward_cost                   ---> 2.098945
    backward_cost                  ---> 0.010771
    deepth                         ---> 0
    collect_flag                   ---> none
    children
        MyNet.linear1
            forward_cost                   ---> 0.041632
            backward_cost                  ---> 0.01486
            deepth                         ---> 1
            collect_flag                   ---> none
        MyNet.linear2
            forward_cost                   ---> 0.62158
            backward_cost                  ---> 0.009877
            deepth                         ---> 1
            collect_flag                   ---> none
        MyNet.linear3
            deepth                         ---> 1
            collect_flag                   ---> list
            children
                MyNet.linear3.0
                    forward_cost                   ---> 0.054222
                    backward_cost                  ---> 0.0484
                    deepth                         ---> 2
                    collect_flag                   ---> none
                MyNet.linear3.1
                    forward_cost                   ---> 0.053606
                    backward_cost                  ---> 0.0509
                    deepth                         ---> 2
                    collect_flag                   ---> none
        ....................
  ```

- You can visulize the cost by 

  ```
  fig,ax = visulize_the_profile_dict(pool,time_flag=['forward_cost'],figsize=(8,8), output_path='test.png')
  ```

  ![](figures\test.png)

**Note**: 
 - we only collect once forward time and backward time. This is not a benchmarking utility like `timeit` or `pytorch.utils.benchmark` which run a piece of code multiple times to capture more accurate timings
 - This module only works that a `nn.Module` return a Tensor of tuple of Tensors, otherwise, you fail to build hood. 
 ```
 WARNING: For backward hooks to be called, module output should be a Tensor or a tuple of Tensors
 ```

#### Installation

```
$ git clone https://github.com/kshitij12345/torchnnprofiler.git
$ cd torchnnprofiler
$ python setup.py install  # Note: It works for pytorch > 2.
```

~~Link to install PyTorch Nightly: https://pytorch.org~~
It is now work for the the pytorch 2

#### visulize arguments
  `time_flag` are using for channel choose. The default is 
  `time_flag='forward_cost'`
  
  It can be `'forward_cost'` , `'backward_cost'`, or the list `['backward_cost','backward_cost']`

  Currently, the `'backward_cost'` cannot successfully record the backward time for the ROOT `nn.Module`, which is the while model. If you have any idea about this, welcome for pull-request or issue.
  
## Motivation

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

However, we want to have a high level profile that record time layer by layer, at least follow the level of `nn.Module`.

# HuggingFace ViT example
<details>
<summary>Profile Code</summary>

```python
import transformers 
def BaseModelOutput(last_hidden_state,hidden_states,attentions,):
    return (last_hidden_state,hidden_states,attentions)
transformers.models.vit.modeling_vit.BaseModelOutput = BaseModelOutput
### The huggingface code wrapper the output into a dataclass named `BaseModelOutput` in ViT.
### Should change it to tuple.

from transformers import ViTConfig, ViTModel
# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()
# Shrink Model, we don't need very large model for demo
configuration.hidden_size = 128
configuration.intermediate_size = 256
configuration.num_hidden_layers = 4
configuration.num_attention_heads = 16
model = ViTModel(configuration)
# Accessing the model configuration
configuration = model.config

# Warm-up
inp= torch.randn(1,3, 224,224).cuda()
model=model.cuda()
out= model(inp,return_dict=False)

with LayerProf(model) as prof:
    model(inp,return_dict=False)[0].sum().backward()
    summary_str = prof.layerwise_summary()
    pool = prof.layerwise_timepool()

# now you get the run-once-profile of this ViT model 
# lets plot it 
# 
fig,ax = visulize_the_profile_dict(pool,time_flag=['forward_cost'],figsize=(16,16), output_path='test.png')
plt.tight_layout() 
```
</details>

**Output:**
![](figures\test_vit.png)
```
**NOTE**: We are unable to capture the timings for `bn` and `RELU` because of inplace operations either performed by the layer or following it.

Ref: https://github.com/pytorch/pytorch/issues/61519

#### IMPORTANT: The hooks mechanism that we utilize for timing the backward pass is only available on the nightly version of PyTorch and will take a few months to be released in the stable version.
```
