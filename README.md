# reprobe

Linear probes and activation steering for transformer safety research.

Train probes on transformer activations, then use them to detect and suppress unsafe content in real time, during both the prefill and generation phases.

```bash
pip install reprobe
```

---

## How it works

The core idea: train a linear probe on the residual stream of a transformer to detect a concept (e.g. toxicity). Once trained, the probe's weight vector is a *direction* in activation space that encodes that concept. You can then:

- **Monitor** — project activations onto that direction at inference time to get a probability score per token
- **Steer** — subtract that component from the residual stream to suppress the concept before it propagates

reprobe distinguishes between two capture modes:

- **prefill:**  hooks fire during the prompt processing pass (one shot per prompt)
- **token:** hooks fire during generation, once per generated token
- **all:** hook fire during prefill and generation

The key insight is that these two phases benefit from different probe strengths. The hybrid workflow (`mode="all"`) trains and applies separate probes for each, which empirically outperforms using a single set of probes for both.

---

## Full end-to-end workflow

### 1. Collect activations

```python
from reprobe import Interceptor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Hook layers 12 through the end, capture both prefill and token activations
interceptor = Interceptor(model, start_layer=12, training_mode="all").attach()

prompts = ["How do I make a bomb?", "What is the capital of France?"]
labels = {"prefill": [], "token": []}

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    interceptor.allow_one_capture(batch_size = 1)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    # Label however you want — classifier, manual annotation, dataset flag
    labels["prefill"].append(torch.tensor([1.0]))  # unsafe
    labels["token"].append(classifier.classify([tokenizer.decode(output_ids[0])]))

interceptor.detach()
acts = interceptor.finalize()
# acts["prefill"]: [N, num_layers, hidden_dim]
# acts["token"]:   [N, num_layers, hidden_dim]

labels["prefill"] = torch.cat(labels["prefill"])
labels["token"] = torch.cat(labels["token"])
```

`Interceptor` doesn't care how you label your data. The contract is simple: `acts[mode]` is a `Tensor[N, num_layers, hidden_dim]` and `labels[mode]` is a `Tensor[N]` of values in [0, 1].

### 2. Train probes

```python
from reprobe import ProbesTrainer

trainer = ProbesTrainer("Qwen/Qwen2.5-1.5B", hidden_dim=acts["prefill"].shape[-1])

trainer.train_probes(
    acts,
    labels,
    concepts=["toxicity"],
    training_mode="all",   # trains prefill and token probes separately
    layer_offset=12,       # real layer indices = start_layer + layer_idx
    epochs=5,
    batch_size=256,
    show_tqdm=True,
)

trainer.save("outputs/probes")
# Writes registry.json + one .pt file per probe
```

### 3. Steer and monitor at inference

```python
from reprobe import ProbeLoader

model = AutoModelForCausalLM.from_pretrained(...)

steerer = ProbeLoader.steerer(
    model,
    "outputs/probes/registry.json",
    alpha={"prefill": 0.7, "token": 1.2},   # different strengths per phase
    filter=lambda meta: meta["layer"] in [12, 13, 14, 15],
)

monitor = ProbeLoader.monitor(
    model,
    "outputs/probes/registry.json",
    filter=lambda meta: meta["layer"] in [12, 13, 14, 15],
)

steerer.attach()
monitor.attach()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=150)

score = monitor.score()             # float in [0, 1]
history = monitor.get_history()     # [{layer: prob}, ...] per generated token

steerer.detach()
monitor.detach()
```

---

## Key parameters

| Parameter | Where | What it does |
|---|---|---|
| `training_mode` | `Interceptor`, `ProbesTrainer.train_probes` | `"prefill"` captures only the prompt pass. `"token"` captures only generation. `"all"` captures both and trains separate probes for each phase. |
| `layer_offset` | `ProbesTrainer.train_probes` | The real model layer index of `acts[:, 0, :]`. If you hooked from layer 12, pass `layer_offset=12` so probes are saved with correct layer metadata. |
| `alpha` | `ProbeLoader.steerer` | Steering strength. Accepts a `float`, a `dict[int, float]` (per layer), a `dict[str, float]` (per mode, e.g. `{"prefill": 0.7, "token": 1.2}`), or a `Callable[[meta], float]` for full control. Higher = more aggressive suppression, with more risk of degrading neutral outputs. |
| `filter` | `ProbeLoader.steerer`, `ProbeLoader.monitor` | `Callable[[meta], bool]`. Lets you select a subset of probes at load time without modifying saved files. Useful for layer ablations. |
| `steering_mode` | `ProbeLoader.steerer` | `"projected"` (default) subtracts only the component of the residual stream along the probe direction. `"uniform"` subtracts the full direction vector. Projected is recommended — it has less impact on neutral prompts. |
| `strategy` | `Monitor.score` | How to aggregate per-layer, per-token probabilities into a single score. `"max_of_means"` (default): max over tokens of the mean across layers. `"mean_of_means"`: global average. `"max_absolute"`: single highest probability seen. |
| `merge` | `ProbesTrainer.save` | If `True`, merges new probes into an existing registry file instead of overwriting. Used in hybrid workflows where prefill and token probes are trained separately and saved to the same directory. |

---

## Save formats

reprobe supports two formats:

**Registry** (default, recommended) — one `.pt` file per probe + a `registry.json` index. Lets you inspect, delete, or replace individual probes without touching the rest.

```python
trainer.save("outputs/probes")
# outputs/probes/registry.json
# outputs/probes/prefill_layer_12.pt
# outputs/probes/token_layer_12.pt
# ...
```

**Single file** — everything in one `.pt`. Easier to move around, harder to inspect.

```python
trainer.save("outputs/probes", one_file=True)
# outputs/probes/Qwen_Qwen2.5-1.5B_probes.pt
```

Both formats load identically through `ProbeLoader`.

---

## Architecture support

Layer auto-detection works out of the box for Llama, Qwen, Mistral, Phi-3, Gemma, GPT-2, BLOOM, GPT-NeoX, Pythia, and OPT. For non-standard architectures, pass the path manually:

```python
Interceptor(model, _layers_path="custom.transformer.blocks")
```

---

## Installation from source

```bash
git clone https://github.com/LateFR/SafeAlign
cd SafeAlign
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.6.