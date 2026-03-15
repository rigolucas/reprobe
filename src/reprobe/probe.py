import json
import os
from typing import Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
import tqdm
import logging

logger = logging.getLogger(__name__)
class ProbesTrainer():
    def __init__(self, model_id: str, hidden_dim: int, device: torch.device = "cpu"):
        self.model_id = model_id
        self.hidden_dim = hidden_dim
        self.probes: dict[int, Probe] = {}
        self.device = device
        
        # defaults, overridables
        self.optim = torch.optim.Adam
        self.optim_kwargs = {"lr": 1e-3, "weight_decay": 1e-4}
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        
        self.num_layers: int = None
        self.layer_offset: int = None  
        
        self.training_mode = None

    def set_optim(self, optim, **kwargs):
        self.optim = optim
        self.optim_kwargs = kwargs
        
        
    def _train_one(
        self,
        probe: "Probe",
        acts: torch.Tensor, # [N, hidden_dim]
        labels: torch.Tensor, # [N, 1]
        epochs: int = 15,
        batch_size: int = 32,
        train_size: float = 0.8, #beetween 0 and 1, test_size will be 1 - train_size
        show_tqdm = False
    ):
        dataset = TensorDataset(acts, labels)
        
        num_samples = len(acts)
        train_n = int(train_size * num_samples)  # train_size = le paramètre float
        test_n = num_samples - train_n
        
        # split
        train_ds, test_ds = random_split(dataset, [train_n, test_n])
        
        train_acts = torch.stack([x for x, _ in train_ds])
        std_act = train_acts.std(dim=0) + 1e-6
        mean_act = train_acts.mean(dim=0)
        probe.mean_act = mean_act
        probe.std_act = std_act

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        optimizer = self.optim(probe.parameters(), **self.optim_kwargs)
        
        probe.train()
        for _ in tqdm.tqdm(range(epochs), desc=f"Probe layer {probe.meta["layer"]}", unit="epoch", disable= not show_tqdm, leave=False):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                loss = self.criterion(probe(batch_X), batch_y)
                loss.backward()
                optimizer.step()

        probe.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)

                logits = probe(batch_X).cpu()
                all_logits.append(logits)
                all_labels.append(batch_y)

        all_logits = torch.cat(all_logits).squeeze()
        all_labels = torch.cat(all_labels).squeeze()

        probs = torch.sigmoid(all_logits)

        auc = roc_auc_score(all_labels.numpy(), probs.numpy())

        return auc
    
    def train_probes(
        self,
        acts: torch.Tensor,    # [N, num_layers, hidden_dim]
        labels: torch.Tensor,  # [N]
        concepts: list[str],
        training_mode: Literal["prefill", "token"]  = "token",
        layer_offset: int = 0,
        epochs: int = 15,
        batch_size: int = 32,
        show_tqdm: bool = False,
        show_stats: bool = True
    ):
        self.training_mode = training_mode
        y = (labels > 0.5).float().unsqueeze(1) # from probability to bool
        
        num_layers = acts.shape[1]
        
        self.num_layers = num_layers
        self.layer_offset = layer_offset
        
        for layer_idx in tqdm.tqdm(range(num_layers), desc="Training Probe", disable=not show_tqdm):
            real_layer = layer_offset + layer_idx
            probe = Probe(
                hidden_dim=self.hidden_dim,
                concepts=concepts,
                layer = real_layer,
                model_id=self.model_id,
                training_mode = training_mode,
            ).to(self.device)
            
            acc = self._train_one(
                probe, 
                acts[:, layer_idx, :],
                y,
                epochs,
                show_tqdm=show_tqdm,
                batch_size=batch_size
            )
            probe.meta["auc"] = round(acc, 4)
            self.probes[real_layer] = probe
            
            if show_stats:
                tqdm.tqdm.write(f"Layer {real_layer} | ROC-AUC: {acc:.3f}")
    
    def save(self, dir: str, one_file: bool = False, merge = False):
        if not self.training_mode:
            raise RuntimeError('Please call "train_probes" before save.')
        os.makedirs(dir, exist_ok=True)
        registry = {
            "model": self.model_id,
            "num_layers": self.num_layers,
            "layer_offset": self.layer_offset,
            "training_mode": self.training_mode,
            "probes": {
                "prefill": {},
                "token": {}
            }
            }
        if not one_file:
            path = os.path.join(dir, "registry.json")
        else:
            path = os.path.join(dir, f"{self.model_id}_probes.pt")
        
        if not os.path.exists(path) and merge:
            logger.warning(f"No probe file existing in {dir}")
        elif merge:
            if not one_file:
                with open(path, "r") as f:
                    registry = json.load(f)
            else:
                registry = torch.load(path)
        if not one_file:
            
            for layer, probe in self.probes.items():
                filename = f"layer_{layer}.pt"
                probe.save(os.path.join(dir, filename))
                registry["probes"][self.training_mode][layer] = {**probe.meta, "filename": filename}
            with open(path, "w") as f:
                json.dump(registry, f, indent=2)
        else:
            for layer, probe in self.probes.items():
                registry["probes"][self.training_mode][layer] = probe._to_save()
            torch.save(registry, path)
        
class Probe(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        concepts: list[str], 
        layer: int, 
        model_id: str,
        training_mode: Literal["prefill", "token"],
        auc: float | None = None,
        mean_act: torch.Tensor | None = None,
        std_act: torch.Tensor | None = None,
    ):
        super().__init__()
        self.meta = {
            "concepts": concepts, # can be anything, just for doc
            "layer": layer,
            "model_id": model_id,
            "training_mode": training_mode,
            "hidden_dim": hidden_dim,
            "auc": auc,
        }
        
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        self.mean_act = mean_act
        self.std_act = std_act
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_act is not None and self.std_act is not None:
            x = (x - self.mean_act.to(x.device)) / self.std_act.to(x.device)
        return self.model(x)
    
    def get_direction(self) -> torch.Tensor:
        # Le vecteur de steering — c'est les weights du Linear
        w = self.model[0].weight.data.squeeze(0)
        return w/ w.norm()
    
    def _to_save(self):
        return {
            "meta": self.meta,
            "state_dict": self.state_dict(),
            "mean_act": self.mean_act,
            "std_act": self.std_act
        }
    def save(self, path: str):
        torch.save(self._to_save(), path)
    
    @classmethod
    def load(cls, state_dict, mean_act: torch.Tensor, std_act: torch.Tensor, **kwargs) -> "Probe":
        probe = cls(
            mean_act = mean_act,
            std_act = std_act,
            **kwargs,
        )
        
        probe.load_state_dict(state_dict)
        
        return probe
    
    @classmethod
    def load_from_file(cls, path: str) -> "Probe":
        checkpoint = torch.load(path)
        meta = checkpoint["meta"]

        probe = cls(mean_act=checkpoint["mean_act"], std_act=checkpoint["std_act"] ,**meta)
        
        probe.load_state_dict(checkpoint["state_dict"])
        
        return probe