import torch
import tempfile, os
from reprobe import Probe

def test_probe_save_and_load():
    # Mock probe
    probe = Probe(hidden_dim=16, concepts=["toxicity"], layer=5, model_id="test", training_mode="prefill")
    probe.mean_act = torch.zeros(16)
    probe.std_act = torch.ones(16)

    # On la sauvegarde dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        probe.training_mode = "prefill"
        probe.save(path)
        loaded = Probe.load_from_file(path)

        # Est-ce que les métadonnées survivent ?
        assert loaded.meta["layer"] == 5
        assert loaded.meta["hidden_dim"] == 16

        # Est-ce que les poids sont identiques ?
        original_w = probe.model[0].weight.data
        loaded_w = loaded.model[0].weight.data
        assert torch.allclose(original_w, loaded_w), "Les poids ont changé après save/load !"
    finally:
        os.remove(path)
        
        
def test_probe_get_direction_guard_zero_norm():
    probe = Probe(hidden_dim=4, concepts=["a"], layer=0, model_id="x", training_mode="token")
    probe.model[0].weight.data.zero_()  # force le cas norm == 0
    d = probe.get_direction()
    assert torch.isfinite(d).all(), "get_direction must not return NaN"
    assert d.shape[0] == 4