from typing import Literal
import torch
from .hook import Hook
from .probe import Probe

class Steerer(Hook):
    def __init__(
            self,
            model,
            probes: list,
            mode: Literal["projected", "uniform"] = "projected",
            alpha: float = 1,
            _layers_path: str | None = None
    ):
        super().__init__(model, _layers_path)
        self.probes: list[tuple[Probe, float]] = [
            (p, alpha) if isinstance(p, Probe) else (p[0], p[1])
            for p in probes
        ]
        self.mode = mode

    def _get_hook(self, layer_idx, data):
        probe, alpha = data
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        direction = probe.get_raw_direction().to(device, dtype)
        mode = probe.meta["training_mode"]
        
        def _hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            is_prefill = hidden.shape[1] > 1
            if mode == "prefill" and not is_prefill:
                return  # ignore generated tokens
            elif mode == "token" and is_prefill:
                return  # ignore prefill
            
            if self.mode == "projected":
                hidden = Steerer._apply_projection(hidden, direction, alpha)
            else:
                hidden = Steerer._apply_uniform(hidden, direction, alpha)
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            
            return hidden
        return _hook_fn

    @staticmethod
    def _apply_projection(hidden, direction, alpha):
        # Scalair product
        # (batch, seq, dim) @ (dim,) -> (batch, seq)
        dot_product = torch.matmul(hidden, direction)
                
        # (batch, seq, 1) * (dim,) -> (batch, seq, dim)
        projection = dot_product.unsqueeze(-1) * direction
        
        hidden = hidden - alpha * projection
        return hidden
    
    @staticmethod
    def _apply_uniform(hidden, direction, alpha):
        return hidden - alpha * direction
    
    def _get_layers_to_hook(self):
        return [(probe.meta["layer"], (probe, alpha)) for probe, alpha in self.probes]
    