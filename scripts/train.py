import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import requests
from tqdm import tqdm
import argparse
import logging
import tomllib
from detoxify import Detoxify
from reprobe import ProbesTrainer, Interceptor, Classifier, ActivationStore
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def ollama_violence_score(texts: list[str], model: str = "gemma3:1b", host: str = "http://localhost:11434") -> torch.Tensor:
    scores = []
    for text in texts:
        prompt = f"""Score the following text on its level of violent intent, from 0.0 (completely safe) to 1.0 (explicitly violent).

Return ONLY a single float between 0.0 and 1.0, nothing else.

Text: {text}

Score:"""
        try:
            response = requests.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            raw = response.json()["message"]["content"].strip()
            score = float(raw)
            score = max(0.0, min(1.0, score))
        except Exception:
            logger.warning(f"Bad answer: {raw}")
            score = 0.0
        scores.append(score)
    
    return torch.tensor(scores)

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", "-c", help="Config path", default="scripts/config.toml")
    
    args = parser.parse_args()
    
    
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    
    
    output_dir = config.get("output", {"output_dir": "output"}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "acts.pt")
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["training"]["device"] and config["training"]["device"] != "auto":
        device = config["training"]["device"]
    
    logger.info(f"Using {device}")
    
    model_id = config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B")
    start_layer_to_hook = config.get("model", {}).get("starting_layer", 1)
    end_layer = config.get("model", {}).get("end_layer", None)
    end_layer = None if end_layer == "last" else end_layer
    max_input_tokens = config.get("max_input_tokens", 2048)
    batch_size = config.get("model", {}).get("batch_size", 1)
    num_samples = config.get("dataset", {}).get("num_samples", 200)

    mode = config["training"].get("mode", "prefill")
    
    dataset_dir = config.get("dataset_gen", {}).get("output_dir", "outputs")
    logger.info(f"Loading tokenizer and model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" #batching
    
    # Chargement en bfloat16 vital pour ta RTX 3060 (12GB)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16,
        device_map=device
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Loading classifier...")
    detox = Detoxify('original', device=device)
    def _normalize(out) -> torch.Tensor:
        TOXIC_BERT_CLASSES = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
        RELEVANT_IDX = [0, 1, 3, 4]  # toxicity, severe_toxicity, threat, insult
        # out.logits : [batch, 6] — pas des probas, des logits bruts
        probs = torch.sigmoid(out.logits)          # → [batch, 6] dans [0,1]
        score = probs[:, RELEVANT_IDX].max(dim=1).values  # → [batch] scalaire par sample
        return score.detach().cpu()
    classifier = Classifier(detox.model, device=device, normalize_output=_normalize, tokenizer=detox.tokenizer)
    
    
    logger.info("Downloading/Loading BeaverTails dataset...")
    # On charge uniquement le split train pour aller vite
    #ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    dataset_path = os.path.join(dataset_dir, f"violence_dataset.pt")
    logger.info(f"Loading dataset from {dataset_path}...")
    raw = torch.load(dataset_path)
    # 2. Filtrage ultra-strict pour la V0.0.1 (On cible la violence/guerre)
    logger.info(f"Filtering dataset: {num_samples} Safe vs {num_samples} Unsafe...")
    safe_data   = [x for x in raw if x["is_safe"]]
    unsafe_data = [x for x in raw if not x["is_safe"]]
    
    # Données "Safe" classiques
    # safe_data = ds.filter(lambda x: x["is_safe"] == True).select(range(num_samples))
    
    # # Données "Unsafe" ciblées uniquement sur la violence (on ignore le reste pour le PoC)
    # unsafe_data = ds.filter(
    #     lambda x: x["is_safe"] == False and (x["category"]["hate_speech,offensive_language"] or x["category"]["terrorism,organized_crime"])
    # ).select(range(num_samples))

    logger.info(f"Safe data: {len(safe_data)}, unsafe data: {len(unsafe_data)}")
    interceptor = Interceptor(model, start_layer_to_hook, end_layer, training_mode=mode).attach()

    store = ActivationStore(
        os.path.join(output_dir, "acts", "acts.h5"),
        len(safe_data) + len(unsafe_data),
        mode,
        start_layer_to_hook,
        len(interceptor._resolve_layers(model))
    )
    # 4. Inférence et Collecte
    labels = {"prefill": []}
    all_texts = {"token": []}

    def process_data(dataset, desc):
        items = list(dataset)
        for i in tqdm(range(0, len(items), batch_size), desc=desc):
            batch = items[i:i+batch_size]
            prompts = [item["prompt"] for item in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,  # obligatoire pour le batch
                max_length=max_input_tokens,
                pad_token_id=tokenizer.pad_token_id
            ).to(device)
            
            interceptor.allow_one_capture(len(batch))
            
            if mode == "prefill":
                with torch.inference_mode():
                    model(**inputs)
                
                acts = interceptor.flush_batch()
                labels = torch.tensor([0.0 if item["is_safe"] else 1.0 for item in batch])
                store.append(
                    acts,
                    {"prefill": labels, "token": None}
                )
                    
            elif mode == "token":
                with torch.inference_mode():
                    model.generate(**inputs, max_new_tokens=100, do_sample=False)
                acts = interceptor.flush_batch()
                
                # prompt_len = inputs["input_ids"].shape[1]
                #texts = tokenizer.batch_decode(outputs_ids[:, prompt_len:], skip_special_tokens=True)
                # labels = [
                #     classifier.classify([text]).squeeze().repeat(prompt_acts.shape[0])
                #     for prompt_acts, text in zip(acts["token"], texts)
                # ]
                token_labels = [
                    torch.tensor(0.0 if item["is_safe"] else 1.0).repeat(prompt_acts.shape[0])
                    for item, prompt_acts in zip(batch, acts["token"])
                ]
                store.append(
                    acts,
                    {"prefill": None, "token": token_labels}
                )
            elif mode == "all":
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                prompt_len = inputs["input_ids"].shape[1]
                texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
                flushed = interceptor.flush_batch()
                
                token_labels = [
                    classifier.classify([text]).squeeze().repeat(prompt_acts.shape[0])
                    for prompt_acts, text in zip(flushed["token"], texts)
                ]
                
                prefill_labels = torch.tensor([0.0 if item["is_safe"] else 1.0 for item in batch])
                token_labels = [
                    torch.tensor(0.0 if item["is_safe"] else 1.0).repeat(prompt_acts.shape[0])
                    for item, prompt_acts in zip(batch, acts["token"])
                ]
                
                store.append(
                    flushed,
                    {"prefill": prefill_labels, "token": token_labels}
                )

    logger.info(f"Batch size set to {batch_size}")
    data = unsafe_data + safe_data
    random.shuffle(data)
    process_data(data, "Extracting acts")
    
    
    interceptor.detach()
            
    logger.info(f"Saving to {output_file}...")
    
    logger.info("Extraction complete. Ready for Supervisor Training!")

    hidden_dim = store.hidden_dim
             
    probe_trainer = ProbesTrainer(config.get("model").get("name", "unknow"), hidden_dim, device="cuda")
    
    probe_trainer.train_probes(
        store,
        concepts=['toxicity', 'severe_toxicity', 'threat', 'insult'],
        epochs=5,
        batch_size= 256,
        show_tqdm = True,
        training_mode=mode
    )
    
    probe_trainer.save(f"{output_dir}/probes")
    
    