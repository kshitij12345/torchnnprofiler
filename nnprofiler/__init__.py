import torch
import datetime
from functools import partial


def get_children(model: torch.nn.Module, name=""):
    children = list(model.named_children())
    flatt_children = []
    # Base case
    if children == []:
        return [
            [name, model],
        ]

    for child_name, child in children:
        if name != "":
            child_name = name + "." + child_name
        flatt_children.extend(get_children(child, child_name))
    return flatt_children


class LayerProf:
    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self.model = model

    def __enter__(self):
        self.layers_event = {}
        self.layers = {}
        self.layer_to_name = {}
        self.layer_device = {}
        self.cnt = 0
        for name, layer in get_children(self.model):
            params = list(layer.parameters())
            if params == []:
                self.layer_device[name] = None
            else:
                self.layer_device[name] = params[0].device
            self.layer_to_name[layer] = name
            self.layers[name] = layer

            def repr_fn(name):
                if not hasattr(self, "layer_times"):
                    return ""

                if name in self.layer_times:
                    times = self.layer_times[name]
                    forward_str = "Forward Time: " + str(times["forward"])
                    backward_str = "Backward Time: " + str(times["backward"])
                    return forward_str + " | " + backward_str

                return ""

            layer.extra_repr = partial(repr_fn, name=name)
            if self.layer_device[name] == torch.device("cuda"):
                self.layers_event[name] = {
                    "forward_pre": torch.cuda.Event(enable_timing=True),
                    "backward_pre": torch.cuda.Event(enable_timing=True),
                    "forward_post": torch.cuda.Event(enable_timing=True),
                    "backward_post": torch.cuda.Event(enable_timing=True),
                }
            else:
                self.layers_event[name] = {
                    "forward_pre": None,
                    "backward_pre": None,
                    "forward_post": None,
                    "backward_post": None,
                }

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        for name, layer in self.layers.items():
            layer.extra_repr = None

    def register_cuda_hooks(self, name):
        events = self.layers_event[name]
        layer: torch.nn.Module = self.layers[name]

        def bw_pre_hook(module, _):
            events["backward_pre"].record()

        def bw_hook(module, _, _1):
            self.cnt += 1
            events["backward_post"].record()

        def fw_pre_hook(module, _):
            events["forward_pre"].record()

        def fw_hook(module, _, _1):
            self.cnt += 1
            events["forward_post"].record()

        layer.register_forward_pre_hook(fw_pre_hook)
        layer.register_forward_hook(fw_hook)

        layer.register_full_backward_pre_hook(bw_pre_hook)
        layer.register_full_backward_hook(bw_hook)

    def register_cpu_hooks(self, name):
        events = self.layers_event[name]
        layer: torch.nn.Module = self.layers[name]

        def bw_pre_hook(module, _):
            events["backward_pre"] = datetime.datetime.now()

        def bw_hook(module, _, _1):
            self.cnt += 1
            events["backward_post"] = datetime.datetime.now()

        def fw_pre_hook(module, _):
            events["forward_pre"] = datetime.datetime.now()

        def fw_hook(module, _, _1):
            self.cnt += 1
            events["forward_post"] = datetime.datetime.now()

        layer.register_forward_pre_hook(fw_pre_hook)
        layer.register_forward_hook(fw_hook)

        layer.register_full_backward_pre_hook(bw_pre_hook)
        layer.register_full_backward_hook(bw_hook)

    def attach_backward_hook(self, name):
        device = self.layer_device[name]
        if device == torch.device("cuda"):
            self.register_cuda_hooks(name)

        self.register_cpu_hooks(name)

    def get_timings(self):
        self.layer_times = {}
        if self.cnt == 0:
            raise RuntimeError(
                "None of the layer recorded time. Did you call forward or backward?"
            )

        for key, value in self.layers_event.items():
            try:
                if self.layer_device[key] == torch.device("cuda"):
                    self.layer_times[key] = {
                        "backward": value["backward_pre"].elapsed_time(
                            value["backward_post"]
                        ),
                        "forward": value["forward_pre"].elapsed_time(
                            value["forward_post"]
                        ),
                    }
                else:
                    self.layer_times[key] = {
                        "backward": (
                            value["backward_post"] - value["backward_pre"]
                        ).total_seconds(),
                        "forward": (
                            value["forward_post"] - value["forward_pre"]
                        ).total_seconds(),
                    }
            except:
                pass

        return self.layer_times
