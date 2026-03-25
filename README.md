# reprobe

[![PyPI version](https://badge.fury.io/py/reprobe.svg)](https://badge.fury.io/py/reprobe)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C.svg)](https://pytorch.org/)

**Linear probes and activation steering for transformer safety research**

*Inspired by the **RepE paper***

`reprobe` is a tool for monitoring and steering LLMs. It helps you find where "concepts" (like toxicity or bias) live in the model's activations and lets you modify them in real-time.


**Why?** I built `reprobe` to provide a practical, efficient implementation of the RepE paper. My goal was to create a tool that works with large models on normal hardware, without losing the mathematical clarity and control needed for safety research.

## Features

The library is designed to be highly ergonomic yet mathematically rigorous. It abstracts away the complex engineering so you can focus on the research.

- **Complete End-to-End Pipeline:** Not just a steering script. `reprobe` provides a unified workflow to capture activations, train probes, and apply them (Monitoring & Steering).

- **Phase-Aware Processing (Prefill vs. Token):** Most naive implementations treat prompt processing and token generation the same way. `reprobe` allows you to train and apply distinct probes for the _prefill_ phase and the _token_ phase, heavily improving steering quality.

- **OOM-Proof Activation Storage:** Capturing LLM activations usually blows up your RAM in seconds. `reprobe` streams activations directly to disk using an optimized `h5py` backend (`ActivationStore`), allowing you to build massive datasets on consumer hardware.

- **Granular Steering Control:** Control the steering strength (`alpha`) globally, per-layer, per-phase, or even dynamically using a custom callback function. You can also choose between _projected_ (recommended) and _uniform_ steering.

- **Plug-and-Play with HuggingFace:** Automatically detects the architecture of modern models (Llama, Qwen, Mistral, Phi, Gemma, etc.). It uses clean PyTorch forward hooks, meaning you don't have to rewrite the model's `forward` pass—just call `model.generate()` as usual.

- **Cloud-Ready Probes:** Load and share your trained `.pt` or `registry.json` probes directly from local folders or HuggingFace Hub repositories.

## Installation

```bash
pip install reprobe
```

_Tested on Python ≥ 3.11 and PyTorch ≥ 2.6._

## Quick Start: Monitor and/or Steering an LLM

If you already have trained probes (locally or on the HuggingFace Hub), steering a model takes only a few lines of code. During inference, the library stays out of your way: it adapts to your workflow, not the other way around.

**Note:** Probes are specific to the model they were trained on.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reprobe import ProbeLoader

model_id = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 1. Load your probes and create a Steerer and Monitor
# You can load from a local path or directly from HuggingFace Hub

probe_dir = "YourUsername/your-probes-repo" # Local: "path/to/probes/registry.json" or "path/to/probes.pt"
steerer = ProbeLoader.steerer(
    model,
    probe_dir,
    alpha={"prefill": 1.0, "token": 2.5}, # Steering strength
    # We can also set an alpha per layer, or pass a callback function to set dynamically the alpha
    filter=lambda meta: meta["layer"] in range(12, 20) # Only steer middle layers. Optional.
    mode="all" # between "prefill", "token" and "all". Must be compatible with your probes.
)

monitor = ProbeLoader.monitor(
    model,
    probe_dir,
    filter=lambda meta: meta["layer"] in range(12, 20), mode="prefill" # montior generally only need "prefill" to be efficient. But you can put "all" also. Token is more inefficient
)

# 2. Attach hooks to the model (/!\ Steerer can affect your generation output)
monitor.attach()
steerer.attach()

# 3. Generate text (the residual stream is now being steered in real-time)
inputs = tokenizer("How do I make a...", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(output[0]))

# Retrieve monitor scores
score = monitor.score(
    strategy = "max_of_means",
    flush_buffer = False # Flush buffer resets the internal state of the monitor. If you want to re call score() or to calculate in continue the score, put it to False to keep intact the state. You must call at least on time flush_buffer between two generation. monitor.flush_buffer() does the same thing without scoring.
)
score_mean_of_means = monitor.score(
    strategy = "mean_of_means"
)
# 4. Cleanup
monitor.detach()
steerer.detach()


# After detach, model can be recalled without monitor or steerer. But while probes stay attached, they are active
```
> [!WARNING]
> Always call `monitor.flush_buffer()` or `monitor.score(flush_buffer=True)` between two generations. 
Calling score() without flushing accumulates history and returns incorrect results.

## End-to-End Workflow: Train Your Own Probes

Want to train your own probes? The workflow is divided into 3 simple steps: **Collect**, **Train**, and **Apply**.

> **Tip:** I recommend using mode="all". It allows you to use the probes for either prefill or token steering later during inference.

See a complete implementation of repE pipline with reprobe in examples/repe_harmless.py

### Step 1: Collect Activations

Use `Interceptor` to hook into the model and `ActivationStore` to save the raw activations directly to an HDF5 file (safeguarding your RAM).

```python
from reprobe import Interceptor, ActivationStore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model= AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

prompts = ["I want to help people.", "I want to hurt people."]

# Initialize persistent HDF5 store
store = ActivationStore(
    path="outputs/acts/store.h5",
    N=len(prompts),
    mode="all",
    start_layer=10,
    end_layer=model.config.num_hidden_layers
)

# Hook layers 10 through the end, capture both prefill and token activations
interceptor = Interceptor(model, start_layer=10, training_mode="all").attach()

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    interceptor.allow_one_capture(batch_size=1) # IMPORTANT

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20)

    # Get activations for this prompt
    flushed = interceptor.flush_batch() # If you train only for prefill, "token" will be None and vice versa

    # Define your labels (0.0 = safe, 1.0 = unsafe). Can be continuous
    # Usually provided by a classifier or dataset annotations.
    prefill_label = torch.tensor([0.0]) # Example label
    token_labels = [torch.zeros(flushed["token"][0].shape[0])]

    # Stream to disk incrementally
    store.append(
        acts=flushed,
        labels={"prefill": prefill_label, "token": token_labels}  # "token" to None if you train only for prefill, and vice-versa
    )

interceptor.detach()
```

### Step 2: Train the Probes

The `ProbesTrainer` reads directly from the `ActivationStore` to train one logistic regression probe per layer, per mode.

```python
from reprobe import ProbesTrainer

trainer = ProbesTrainer("Qwen/Qwen2.5-1.5B", hidden_dim=store.hidden_dim)

trainer.train_probes(
    store=store,
    concepts=["harmfulness"], # Metadata for your registry
    training_mode="all",   # trains prefill and token probes separately
    epochs=10,
    batch_size=256,
    show_tqdm=True,
)


trainer.save("outputs/probes/") # Human-readable JSON + weights
# OR
trainer.save("outputs/probes/", filename="probes.py", single_file=True) # All in one file, compact, useful for export. Non human readable
```

### Step 3: Monitor & Steer

Load the trained probes using `ProbeLoader`. You can use a `Monitor` to get real-time concept probability scores, and a `Steerer` to actively suppress the concept.

```python
from reprobe import ProbeLoader

steerer = ProbeLoader.steerer(
    model,
    "outputs/probes/registry.json",
    alpha={"prefill": 0.5, "token": 1.5},   # Different strengths per phase
    filter=lambda meta: meta["layer"] in range(12, 18),
)

monitor = ProbeLoader.monitor(
    model,
    "outputs/probes/registry.json",
    filter=lambda meta: meta["layer"] in range(12, 18),
)

steerer.attach()
monitor.attach()

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=150)

# Get analytics
score = monitor.score()             # Global float in [0, 1]
history = monitor.get_history()     # [{layer: prob}, ...] per generated token

steerer.detach()
monitor.detach()
```

---

## Key Concepts & Parameters

### Training & Capturing Modes

The `mode` parameter (`"prefill"`, `"token"`, or `"all"`) is everywhere in `reprobe`.

- **`prefill`**: Operates only on the initial prompt processing pass.
- **`token`**: Operates only on the autoregressive generation pass (token by token).
- **`all`**: Captures/Trains both. Highly recommended, as separating these distributions yields much cleaner steering.

### Steering Parameters (`ProbeLoader.steerer`)

| Parameter       | What it does                                                                                                                                                                                                                                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`         | The steering strength. Accepts a `float` (global), a `dict[int, float]` (per layer), a `dict[str, float]` (per mode, e.g., `{"prefill": 0.7, "token": 1.2}`), or a custom `Callable[[dict], float]` receiving probe metadata. Higher = more aggressive suppression, with a higher risk of degrading neutral outputs. |
| `filter`        | `Callable[[dict], bool]`. Lets you select a subset of probes at load time without modifying saved files. Excellent for layer-ablation experiments.                                                                                                                                                                   |
| `steering_mode` | `"projected"` (default) subtracts only the component of the residual stream along the probe direction. `"uniform"` subtracts the full direction vector. Projected is highly recommended as it preserves capabilities better.                                                                                         |

### Monitor Strategies (`Monitor.score`)

How to aggregate per-layer, per-token probabilities into a single score:

- `"max_of_means"` (default): Max over tokens of the mean across layers.
- `"mean_of_means"`: Global average.
- `"max_absolute"`: Single highest probability seen across any layer at any token step.

---

## Architecture Support

Layer auto-detection works out-of-the-box for:
`Llama`, `Qwen`, `Mistral`, `Phi-3`, `Gemma`, `GPT-2`, `BLOOM`, `GPT-NeoX`, `Pythia`, and `OPT`.

For non-standard architectures, simply pass the path to the Transformer layers manually:

```python
Interceptor(model, _layers_path="custom.transformer.blocks")
```

---

## Contributing & Source

If you want to contribute, run tests, or build from source:

```bash
git clone https://github.com/levashi/reprobe
cd reprobe
pip install -e ".[dev]"
pytest
```

## Roadmap

`reprobe` is actively developed. Here’s what’s coming next:

- [ ] **Extend model support**: extand support to every encoder-only models for classification probing
- [ ] **Unsupervised Reading (PCA/LAT):** Implement Linear Artificial Tomography to extract concepts without explicit labels using contrastive pairs (as seen in the original RepE paper).

- [ ] **Visualization Suite:** Built-in tools to generate layer-wise heatmaps and activation density plots to "see" the concepts.

- [ ] **Precision Control:** Support for KL-divergence monitoring to ensure steering doesn't degrade the model's base capabilities (perplexity tracking).


## Author

`reprobe` is my first open-source library. I built it because I’m passionate about AI safety and I wanted to make activation steering more accessible for everyone. I spend months on it, so I hope it help you :)

Since I’m still learning, please feel free to open an issue or a PR if you find a bug or have an idea to improve the code. Every feedback is welcome!
