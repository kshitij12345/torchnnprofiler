import torch
import time


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


class CPUEvent:
    def __init__(self):
        self.event_time = None

    def record(self):
        self.event_time = time.process_time_ns()

    def elapsed_time(self, other):
        assert other.event_time > self.event_time
        return (other.event_time - self.event_time) * 1e-6
