import torch
import tempfile, os
from reprobe import Probe
from reprobe import ProbeLoader
from reprobe import ProbesTrainer

def test_probe_save_and_load():
    # Mock probe
    probe = Probe(hidden_dim=16, concepts=["toxicity"], layer=5, model_id="test", training_mode="prefill")
    probe.mean_act = torch.zeros(16)
    probe.std_act = torch.ones(16)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        probe.training_mode = "prefill"
        probe.save(path)
        loaded = Probe.load_from_file(path)

        # metadata?
        assert loaded.meta["layer"] == 5
        assert loaded.meta["hidden_dim"] == 16

        # identical weights?
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

def test_probe_get_raw_direction():
    hidden_dim = 4
    probe = Probe(hidden_dim=hidden_dim, concepts=["a"], layer=0, model_id="x", training_mode="prefill")
    probe.mean_act = torch.zeros(hidden_dim)
    
    # uniform std_act
    probe.std_act = torch.ones(hidden_dim)
    assert torch.allclose(probe.get_direction(), probe.get_raw_direction()) 
    
    assert torch.allclose(probe.get_raw_direction().norm(), torch.tensor(1.0), atol=1e-5) 
    
    # variable std_act
    probe.std_act = torch.tensor([1.0, 2.0, 0.5, 1.0])
    assert not torch.allclose(probe.get_direction(), probe.get_raw_direction()), "std_act must affect get_raw_direction"
    
    assert torch.allclose(probe.get_raw_direction().norm(), torch.tensor(1.0), atol=1e-5) 
    
def test_save_merge():
    hidden_dim = 16

    def make_probe(mode, layer):
        p = Probe(hidden_dim=hidden_dim, concepts=["toxicity"], layer=layer, model_id="test", training_mode=mode)
        p.mean_act = torch.zeros(hidden_dim)
        p.std_act = torch.ones(hidden_dim)
        return p

    trainer_p = ProbesTrainer("test", hidden_dim)
    trainer_p.training_mode = "prefill"
    trainer_p.num_layers = 1
    trainer_p.layer_offset = 5
    trainer_p.probes["prefill"][5] = make_probe("prefill", 5)

    trainer_t = ProbesTrainer("test", hidden_dim)
    trainer_t.training_mode = "token"
    trainer_t.num_layers = 1
    trainer_t.layer_offset = 5
    trainer_t.probes["token"][5] = make_probe("token", 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_p.save(tmpdir)
        trainer_t.save(tmpdir, merge=True)

        loaded = ProbeLoader.from_registry(os.path.join(tmpdir, "registry.json"))
        assert 5 in loaded["prefill"]
        assert 5 in loaded["token"]