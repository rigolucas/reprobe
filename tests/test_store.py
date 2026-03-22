import tempfile

import pytest
import torch

from reprobe import ActivationStore


def test_append_and_get_layer():
    with tempfile.TemporaryDirectory() as dir:
        start_layer = 12
        end_layer = 13
        store = ActivationStore(dir + "test.h5", N=10, mode="all", start_layer=start_layer, end_layer=end_layer)
        
        # prefill : 4 prompts, 2 layers, hidden_dim=8
        acts_p = torch.randn(4, 2, 8)
        labels_p = torch.tensor([1., 0., 1., 0.])

        acts_t = [torch.randn(3, 2, 8), torch.randn(2, 2, 8)]
        labels_t = [torch.ones(3), torch.zeros(2)]

        store.append({"prefill": acts_p, "token": acts_t}, {"prefill": labels_p, "token": labels_t})

        a, l = store.get_layer("prefill", 12)
        assert a.shape == (4, 8)
        assert torch.allclose(a, acts_p[:, 0, :]) 

        a, l = store.get_layer("token", 12)
        assert a.shape == (2, 8)  
        
def test_resume_cursor_integrity():
    with tempfile.TemporaryDirectory() as dir:
        store_path = dir + "/test.h5"
        store1 = ActivationStore(store_path, N=100, mode="prefill", start_layer=0, end_layer=2)
        acts1 = torch.randn(5, 2, 8)
        store1.append({"prefill": acts1}, {"prefill": torch.ones(5)})
        
        del store1 #crash
        
        store2 = ActivationStore(store_path, N=20, mode="prefill", start_layer=0, end_layer=2, resume=True)
        assert store2.cursors["prefill"] == 5
        assert store2.N == 100
        acts2 = torch.randn(5, 2, 8)
        store2.append({"prefill": acts2}, {"prefill": torch.ones(5)})

        a, l = store2.get_layer("prefill", 0)
        assert a.shape == (10, 8)
        assert torch.allclose(a[:5], acts1[:, 0, :])  # les données d'avant sont intactes
        assert torch.allclose(a[5:], acts2[:, 0, :])
        
        with pytest.raises(ValueError):
            ActivationStore(store_path, N=20, mode="token", start_layer=0, end_layer=2, resume=True)
            
            
def test_prefill_overflow_raises():
    with tempfile.TemporaryDirectory() as dir:
        store_path = dir + "/test.h5"
        store = ActivationStore(store_path, N=4, mode="prefill", start_layer=0, end_layer=1)
        store.append({"prefill": torch.randn(4, 1, 8)}, {"prefill": torch.ones(4)})

        with pytest.raises(ValueError):
            store.append({"prefill": torch.randn(1, 1, 8)}, {"prefill": torch.ones(1)})

        a, l = store.get_layer("prefill", 0)
        assert a.shape == (4, 8)
                