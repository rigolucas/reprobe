import logging
from .steerer import Steerer
from .monitor import Monitor
from .probe import Probe
from pathlib import Path
from typing import Callable, Literal
import json
import os
import torch

logger = logging.getLogger(__name__)
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
        probes = {
            "prefill": {},
            "token": {}
        }
        training_mode = registry["training_mode"]
        if not training_mode or training_mode not in ["prefill", "token", "all"]:
                raise ValueError(f"Registry has an invalid mode: {training_mode}. Must be between 'token', 'prefill' and 'all'")
                
        for mode in ["prefill", "token"]:
            for key, meta in registry["probes"][mode].items():
                probe_path = os.path.join(dir, meta["filename"])
                probe = Probe.load_from_file(probe_path)
                probe_mode = probe.meta["training_mode"]
                if not probe_mode or probe_mode not in ["prefill", "token"]:
                    logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {probe_mode}. Must be between 'token' and 'prefill'. Skipped")
                    continue
                probes[mode][probe.meta["layer"]] = probe
                
        return probes
    
    @staticmethod
    def from_file(path: str) -> dict[int, Probe]:
        """
        Load from a .pt file
        Return {layer: Probe}
        """
        content = torch.load(path)
        probes = {
            "prefill": {},
            "token": {}
        }
        for mode in ["prefill", "token"]:
            for key, data in content["probes"][mode].items():
                probe = Probe.load(
                    data["state_dict"],
                    mean_act=data["mean_act"],
                    std_act=data["std_act"],
                    **data["meta"]
                )
                probe_mode = probe.meta["training_mode"]
                if not probe_mode or probe_mode not in ["prefill", "token"]:
                    logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {probe_mode}. Must be between 'token' and 'prefill'. Skipped")
                    continue
                probes[mode][probe.meta["layer"]] = probe
                
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
    def _check_mode(
        mode: Literal["prefill", "token", "all", "auto"],
        probes: dict[str, Probe],
        return_flatten_probes = False
    ) -> dict[str, Probe] | list[Probe]:
        
        if mode in ["prefill", "token"]:
            if not probes[mode]:
                raise ValueError(f"The probes given are not compatible with the mode {mode}. Probes provided has the following keys: {probes.keys()}")
            if return_flatten_probes:
                return list(probes[mode].values())
            
        elif mode == "all":
            if (not "prefill" in probes.keys()) or (not "token" in probes.keys()):
                raise ValueError(f"The probes given are not compatible with the mode {mode}. Probes provided has the following keys: {probes.keys()}")
            if return_flatten_probes:
                return list(probes["prefill"].values()) + list(probes["token"].values())
            
        elif mode == "auto":
            probe_list = list(probes["prefill"].values()) + list(probes["token"].values())
            if not probe_list:
                raise ValueError("No probes found")
            if return_flatten_probes:
                return probe_list
            
        else:
            raise ValueError("Invalid mode")
        
        return probes # if return_flatten_probes=False
            
    @staticmethod
    def monitor(
        model, 
        path: str,
        mode: Literal["prefill", "token", "all", "auto"] = "auto",
        filter: Callable[[dict], bool] = None
    ):
        probes = ProbeLoader.load(path)
        probes = ProbeLoader._check_mode(mode, probes, return_flatten_probes=True)
        if filter:
            probes = [p for p in probes if filter(p.meta)]
        
        return Monitor(model, probes)
    
    @staticmethod
    def steerer(
        model,
        path: str,
        mode: Literal["prefill", "token", "all", "auto"] = "auto",
        steering_mode: Literal['projected', 'uniform'] = "projected",
        alpha: float | dict[int, float] | dict[str, float] | Callable[[dict], float] = 1.0,
        filter: Callable[[dict], bool] = None,
    ):
        probes = ProbeLoader.load(path)
        probes = ProbeLoader._check_mode(mode, probes, return_flatten_probes=True)
        if filter:
            probes = [p for p in probes if filter(p.meta)]
        
        if callable(alpha):
            probe_list = [(p, alpha(p.meta)) for p in probes]
            
        elif isinstance(alpha, dict):
            first_key = next(iter(alpha))
            if isinstance(first_key, str):  # mode dict
                probe_list = [(p, alpha.get(p.meta["training_mode"], 1.0)) for p in probes]
            else:  # layer dict
                probe_list = [(p, alpha.get(p.meta["layer"], 1.0)) for p in probes]
                
        else:
            probe_list = probes #Steerer will automatically add alpha
            
        return Steerer(model, probe_list, mode=steering_mode, alpha=alpha) #alpha will be automatically ignored in the 2 first case
    
    