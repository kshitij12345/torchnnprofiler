import torch
import torch.nn as nn
from functools import partial
import itertools
from .utils import get_children, CPUEvent
import traceback
import time

def get_device_event(device: torch.device):
    if device == torch.device("cuda"):
        return torch.cuda.Event(enable_timing=True)
    return CPUEvent()


class LayerProf:
    def __init__(self, model, profile_all_layers=True, debug=False):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.profile_all_layers = profile_all_layers
        self.in_context_mode = False
        self._layers_event = {}
        self._layers = {}
        self.cnt = 0
        self._hook_handles = []
        self.debug = debug
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

        self._hook_handles.append(
            layer.register_full_backward_pre_hook(bw_pre_hook))
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
                    "backward": value["backward_pre"].elapsed_time(value["backward_post"]
                                                                   ),
                    "forward": value["forward_pre"].elapsed_time(value["forward_post"]),
                }
            except Exception as e:
                if key == self.model._get_name():
                    print(
                        f'The root model should return a tensor for timing. Please modify. If you cannot locate the escape reason, use debug=True')

                if self.debug:
                    print(f'pass {key}')
                    traceback.print_exc()
                pass

        return self.layer_times

    def layerwise_summary(self, precision=6):
        if not hasattr(self, "layers"):
            self.get_timings()

        prev_objs = {}
        prev_name = {}
        #print(self._layers.keys())
        for name, layer in self._layers.items():
            # Hacky stuff below!
            highlevellayer = len(list(layer.named_children())) > 0

            def _get_name(name, model):
                ttt = model.__class__.__name__
                if not hasattr(self, "layer_times"):
                    return ttt

                if name in self.layer_times:
                    times = self.layer_times[name]
                    ftime = times["forward"]
                    btime = times["backward"]
                    forward_str = f"Forward Time: {ftime:.{precision}f}ms"
                    backward_str = f"Backward Time: {btime:.{precision}f}ms"
                    return f"{ttt}【{forward_str}|{backward_str}】"

                return ttt

            def repr_fn(name):
                if not hasattr(self, "layer_times"):
                    return ""

                if name in self.layer_times:
                    times = self.layer_times[name]
                    ftime = times["forward"]
                    btime = times["backward"]
                    forward_str = f"Forward Time: {ftime:.{precision}f}ms"
                    backward_str = f"Backward Time: {btime:.{precision}f}ms"
                    return f"{forward_str}|{backward_str}"

                return ""
            if highlevellayer:

                prev_name[layer] = layer._get_name
                layer._get_name = partial(_get_name, name=name, model=layer)
            else:

                prev_objs[layer] = layer.extra_repr
                layer.extra_repr = partial(repr_fn, name=name)

        return_str = self.model.__repr__()
        for name, layer in self._layers.items():
            if layer in prev_objs:
                layer.extra_repr = prev_objs[layer]
            if layer in prev_name:
                layer._get_name = prev_name[layer]

        return return_str

    def layerwise_timepool(self, precision=6):
        def repr_fn(name):
            if not hasattr(self, "layer_times"):
                return {}

            if name in self.layer_times:
                times = self.layer_times[name]
                ftime = times["forward"]
                btime = times["backward"]
                return {'forward_cost': ftime, 'backward_cost': btime}

            return {}

        def get_children_pool(model: torch.nn.Module, name=None, depth=0):
            if name is None:
                name = model._get_name()
            children = list(model.named_children())
            module_dict = {name: repr_fn(name)}
            module_dict[name]['deepth'] = depth
            if isinstance(model, (nn.ModuleList)):
                module_dict[name]['collect_flag'] = 'list'
            elif isinstance(model, (nn.ModuleDict)):
                module_dict[name]['collect_flag'] = 'dict'
            else:
                module_dict[name]['collect_flag'] = 'none'

            # Base case
            if children == []:
                pass
            else:
                for child_name, child in children:
                    if name != "":
                        child_name = name + "." + child_name
                    if 'children' not in module_dict[name]:
                        module_dict[name]['children'] = {}
                    module_dict[name]['children'].update(
                        get_children_pool(child, child_name, depth+1))
            return module_dict
        return get_children_pool(self.model)
