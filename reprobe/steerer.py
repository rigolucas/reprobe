import torch
from .hook import Hook
from .probe import Probe

class Steerer(Hook):
    def __init__(self, model, probes: list, alpha: float = 20.0):
        super().__init__(model)
        self.probes: list[tuple[Probe, float]] = [
            (p, alpha) if isinstance(p, Probe) else (p[0], p[1])
            for p in probes
        ]

    def _get_hook(self, layer_idx, data):
        probe, alpha = data
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        direction = probe.get_direction().to(device, dtype)
        
        def _hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                
                # Scalair product
                # (batch, seq, dim) @ (dim,) -> (batch, seq)
                dot_product = torch.matmul(hidden, direction)
                
                # (batch, seq, 1) * (dim,) -> (batch, seq, dim)
                projection = dot_product.unsqueeze(-1) * direction
                
                hidden = hidden - alpha * direction
                return (hidden,) + output[1:]
            else:
                dot_product = torch.matmul(output, direction)
                projection = dot_product.unsqueeze(-1) * direction
                return output - alpha * projection
        return _hook_fn

    def _get_layers_to_hook(self):
        return [(probe.meta["layer"], (probe, alpha)) for probe, alpha in self.probes]
    