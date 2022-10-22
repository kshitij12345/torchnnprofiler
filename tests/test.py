import pytest

import torch
import nnprofiler

def test_throws_error_without_ctx_manager():
    prof = nnprofiler.LayerProf(torch.nn.Sigmoid())
    with pytest.raises(RuntimeError, match='as a context-manager'):
        prof.get_timings()
