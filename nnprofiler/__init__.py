import torch
from functools import partial
import itertools
from .utils import get_children, CPUEvent


def get_device_event(device: torch.device):
    if device == torch.device("cuda"):
        return torch.cuda.Event(enable_timing=True)
    return CPUEvent()


class LayerProf:
    def __init__(self, model, profile_all_layers=True):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.profile_all_layers = profile_all_layers
        self.in_context_mode = False
        self._layers_event = {}
        self._layers = {}
        self.cnt = 0
        self._hook_handles = []

        for name, layer in get_children(self.model):
            self._layers[name] = layer

            params_slice = list(itertools.islice(layer.parameters(), 1))
            if params_slice == []:
                layer_device = None
            else:
                layer_device = params_slice[0].device

            # TODO: Only create event for layer if
            # it is going to be profiled!
            self._layers_event[name] = {
                "forward_pre": get_device_event(layer_device),
                "backward_pre": get_device_event(layer_device),
                "forward_post": get_device_event(layer_device),
                "backward_post": get_device_event(layer_device),
            }

            if self.profile_all_layers:
                self._attach_backward_hook(name)

    def _throw_if_not_in_context_mode(self):
        if not self.in_context_mode:
            msg = "You should call the nnprofiler.LayerProf as a context-manager using `with`"
            raise RuntimeError(msg)

    def __enter__(self):
        self.in_context_mode = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        for handle in self._hook_handles:
            # Remove all the hooks that we attached!
            handle.remove()

    def _register_hooks(self, name):
        events = self._layers_event[name]
        layer: torch.nn.Module = self._layers[name]

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

        self._hook_handles.append(layer.register_forward_pre_hook(fw_pre_hook))
        self._hook_handles.append(layer.register_forward_hook(fw_hook))

        self._hook_handles.append(layer.register_full_backward_pre_hook(bw_pre_hook))
        self._hook_handles.append(layer.register_full_backward_hook(bw_hook))

    def _attach_backward_hook(self, name):
        self._register_hooks(name)

    def attach_backward_hook(self, name):
        msg = "Manually attaching hook is only allowed when profile_all_layers is False"
        assert self.profile_all_layers is False, msg
        self._throw_if_not_in_context_mode()
        self._register_hooks(name)

    def get_timings(self):
        self._throw_if_not_in_context_mode()
        self.layer_times = {}
        if self.cnt == 0:
            raise RuntimeError(
                "None of the layer recorded time. Did you call forward or backward?"
            )

        for key, value in self._layers_event.items():
            try:
                self.layer_times[key] = {
                    "backward": value["backward_pre"].elapsed_time(
                        value["backward_post"]
                    ),
                    "forward": value["forward_pre"].elapsed_time(value["forward_post"]),
                }
            except Exception as e:
                # TODO: Handle these better!
                pass

        return self.layer_times

    def layerwise_summary(self, precision=6):
        if not hasattr(self, "layers"):
            self.get_timings()

        prev_objs = {}
        for name, layer in self._layers.items():

            def repr_fn(name):
                if not hasattr(self, "layer_times"):
                    return ""

                if name in self.layer_times:
                    times = self.layer_times[name]
                    ftime = times["forward"]
                    btime = times["backward"]
                    forward_str = f"Forward Time: {ftime:.{precision}f}ms"
                    backward_str = f"Backward Time: {btime:.{precision}f}ms"
                    return forward_str + " | " + backward_str

                return ""

            prev_objs[layer] = layer.extra_repr
            layer.extra_repr = partial(repr_fn, name=name)

        return_str = self.model.__repr__()

        for name, layer in self._layers.items():
            layer.extra_repr = prev_objs[layer]

        return return_str
