from .steerer import Steerer
from .monitor import Monitor
from .probe import Probe
from pathlib import Path
from typing import Callable, Literal
import json
import os
import torch

class ProbeLoader:
    @staticmethod
    def from_registry(path: str) -> dict[int, Probe]:
        """
        Load from a registry.json
        Return {layer: Probe}
        """
        with open(path, "r") as f:
            registry = json.load(f)
        
        dir = os.path.dirname(path)
        probes = {}
        for key, meta in registry.items():
            if not key.isdigit(): # skip global keys
                continue
            
            probe_path = os.path.join(dir, meta["filename"])
            probe = Probe.load_from_file(probe_path)
            probes[probe.meta["layer"]] = probe
            
        return probes
    
    @staticmethod
    def from_file(path: str) -> dict[int, Probe]:
        """
        Load from a .pt file
        Return {layer: Probe}
        """
        content = torch.load(path)
        probes = {}
        for key, data in content.items():
            probe = Probe.load(
                data["state_dict"],
                mean_act=data["mean_act"],
                std_act=data["std_act"],
                **data["meta"]
            )
            probes[probe.meta["layer"]] = probe
            
        return probes
    
    @staticmethod
    def load(path: str) -> dict[int, Probe]:
        path = Path(path)

        if path.suffix == ".pt":
            return ProbeLoader.from_file(path)

        if path.suffix == ".json":
            return ProbeLoader.from_registry(path)

        raise ValueError(f"Unsupported file type: {path}")
    
    
    @staticmethod
    def monitor(
        model, 
        path: str,
        filter: Callable[[dict], bool] = None
    ):
        probes = ProbeLoader.load(path)
        if filter:
            probes = {k: v for k, v in probes.items() if filter(v.meta)}
        
        return Monitor(model, list(probes.values()))
    
    @staticmethod
    def steerer(
        model,
        path: str,
        mode: Literal['projected', 'uniform'] = "projected",
        alpha: float | dict[int, float] | Callable[[dict], float] = 1.0,
        filter: Callable[[dict], bool] = None,
    ):
        probes = ProbeLoader.load(path)
        if filter:
            probes = {k: v for k, v in probes.items() if filter(v.meta)}
        
        if callable(alpha):
            probe_list = [(p, alpha(p.meta)) for p in probes.values()]
        elif isinstance(alpha, dict):
            probe_list = [(p, alpha.get(layer, 20.0)) for layer, p in probes.items()]
        else:
            probe_list = list(probes.values())
            
        return Steerer(model, probe_list, mode=mode, alpha=alpha) #alpha will be automatically ignored in the 2 first case