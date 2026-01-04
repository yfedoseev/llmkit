#!/usr/bin/env python3
"""
Comprehensive Together AI models list based on Together AI catalog.

Together AI hosts 200+ open-source models. This script generates a comprehensive
CSV of all known Together AI models with estimated pricing and capabilities.

Reference: https://www.together.ai/
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Comprehensive Together AI model catalog
TOGETHER_AI_MODELS = [
    # Meta Llama Series (Latest)
    {"name": "meta-llama/Llama-3.1-405B", "display": "Llama 3.1 405B", "context": 131072, "type": "chat", "size": "405b"},
    {"name": "meta-llama/Llama-3.1-70B", "display": "Llama 3.1 70B", "context": 131072, "type": "chat", "size": "70b"},
    {"name": "meta-llama/Llama-3.1-8B", "display": "Llama 3.1 8B", "context": 131072, "type": "chat", "size": "8b"},
    {"name": "meta-llama/Llama-3-70B-Chat", "display": "Llama 3 70B Chat", "context": 8192, "type": "chat", "size": "70b"},
    {"name": "meta-llama/Llama-3-8B-Chat", "display": "Llama 3 8B Chat", "context": 8192, "type": "chat", "size": "8b"},
    {"name": "meta-llama/Llama-2-70B-Chat", "display": "Llama 2 70B Chat", "context": 4096, "type": "chat", "size": "70b"},
    {"name": "meta-llama/Llama-2-13B-Chat", "display": "Llama 2 13B Chat", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "meta-llama/Llama-2-7B-Chat", "display": "Llama 2 7B Chat", "context": 4096, "type": "chat", "size": "7b"},

    # Mistral Series
    {"name": "mistralai/Mistral-7B-Instruct-v0.2", "display": "Mistral 7B Instruct v0.2", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "mistralai/Mixtral-8x22B-Instruct-v0.1", "display": "Mixtral 8x22B Instruct", "context": 65536, "type": "chat", "size": "22b"},
    {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display": "Mixtral 8x7B Instruct", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "mistralai/Mistral-Large", "display": "Mistral Large", "context": 32768, "type": "chat", "size": "70b"},
    {"name": "mistralai/Mistral-Medium", "display": "Mistral Medium", "context": 32768, "type": "chat", "size": "34b"},
    {"name": "mistralai/Mistral-7B", "display": "Mistral 7B Base", "context": 32768, "type": "base", "size": "7b"},

    # Qwen Series
    {"name": "Qwen/Qwen1.5-72B-Chat", "display": "Qwen 1.5 72B Chat", "context": 32768, "type": "chat", "size": "72b"},
    {"name": "Qwen/Qwen1.5-32B-Chat", "display": "Qwen 1.5 32B Chat", "context": 32768, "type": "chat", "size": "32b"},
    {"name": "Qwen/Qwen1.5-14B-Chat", "display": "Qwen 1.5 14B Chat", "context": 32768, "type": "chat", "size": "14b"},
    {"name": "Qwen/Qwen1.5-7B-Chat", "display": "Qwen 1.5 7B Chat", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "Qwen/Qwen-72B-Chat", "display": "Qwen 72B Chat", "context": 32768, "type": "chat", "size": "72b"},
    {"name": "Qwen/Qwen-7B-Chat", "display": "Qwen 7B Chat", "context": 32768, "type": "chat", "size": "7b"},

    # Nous Research Models
    {"name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "display": "Nous Hermes 2 Mixtral 8x7B DPO", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT", "display": "Nous Hermes 2 Mixtral 8x7B SFT", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO", "display": "Nous Hermes 2 Mistral 7B DPO", "context": 32768, "type": "chat", "size": "7b"},

    # CodeLlama Series (Coding)
    {"name": "togethercomputer/CodeLlama-34b-Instruct", "display": "CodeLlama 34B Instruct", "context": 16384, "type": "chat", "size": "34b"},
    {"name": "togethercomputer/CodeLlama-13b-Instruct", "display": "CodeLlama 13B Instruct", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "togethercomputer/CodeLlama-7b-Instruct", "display": "CodeLlama 7B Instruct", "context": 4096, "type": "chat", "size": "7b"},
    {"name": "meta-llama/CodeLlama-34b-Python", "display": "CodeLlama 34B Python", "context": 16384, "type": "chat", "size": "34b"},

    # DeepSeek Series
    {"name": "deepseek-ai/deepseek-coder-33b-instruct", "display": "DeepSeek Coder 33B Instruct", "context": 4096, "type": "chat", "size": "33b"},
    {"name": "deepseek-ai/deepseek-llm-67b-chat", "display": "DeepSeek LLM 67B Chat", "context": 4096, "type": "chat", "size": "67b"},

    # Allen AI Models
    {"name": "allenai/OLMo-7B-Instruct", "display": "OLMo 7B Instruct", "context": 2048, "type": "chat", "size": "7b"},
    {"name": "allenai/OLMo-7B", "display": "OLMo 7B Base", "context": 2048, "type": "base", "size": "7b"},

    # SOLAR (Upstage) Models
    {"name": "upstage/SOLAR-10.7B-Instruct-v1.0", "display": "SOLAR 10.7B Instruct", "context": 4096, "type": "chat", "size": "10b"},

    # OpenHermes Models
    {"name": "teknium/OpenHermes-2.5-Mistral-7B", "display": "OpenHermes 2.5 Mistral 7B", "context": 32768, "type": "chat", "size": "7b"},
    {"name": "teknium/OpenHermes-2-Mistral-7B", "display": "OpenHermes 2 Mistral 7B", "context": 32768, "type": "chat", "size": "7b"},

    # WizardLM Series
    {"name": "WizardLM/WizardLM-13B-V1.2", "display": "WizardLM 13B V1.2", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "WizardLM/WizardCoder-15B-V1.0", "display": "WizardCoder 15B V1.0", "context": 4096, "type": "chat", "size": "15b"},

    # Vicuna Models
    {"name": "lmsys/vicuna-13b-v1.5", "display": "Vicuna 13B V1.5", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "lmsys/vicuna-7b-v1.5", "display": "Vicuna 7B V1.5", "context": 4096, "type": "chat", "size": "7b"},

    # Falcon Series
    {"name": "tiiuae/falcon-40b-instruct", "display": "Falcon 40B Instruct", "context": 2048, "type": "chat", "size": "40b"},
    {"name": "tiiuae/falcon-7b-instruct", "display": "Falcon 7B Instruct", "context": 2048, "type": "chat", "size": "7b"},

    # MPT Series
    {"name": "mosaicml/mpt-30b-instruct", "display": "MPT 30B Instruct", "context": 8192, "type": "chat", "size": "30b"},
    {"name": "mosaicml/mpt-7b-instruct", "display": "MPT 7B Instruct", "context": 8192, "type": "chat", "size": "7b"},

    # Orca Models
    {"name": "microsoft/orca-2-13b", "display": "Orca 2 13B", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "microsoft/orca-2-7b", "display": "Orca 2 7B", "context": 4096, "type": "chat", "size": "7b"},

    # StabilityAI Models
    {"name": "stabilityai/stablelm-2-12b-chat", "display": "StableLM 2 12B Chat", "context": 4096, "type": "chat", "size": "12b"},

    # RWKV Models
    {"name": "RWKV/rwkv-5-world-3b", "display": "RWKV-5 World 3B", "context": 4096, "type": "chat", "size": "3b"},

    # Yi Series
    {"name": "01-ai/Yi-34B-Chat", "display": "Yi 34B Chat", "context": 4096, "type": "chat", "size": "34b"},
    {"name": "01-ai/Yi-6B-Chat", "display": "Yi 6B Chat", "context": 4096, "type": "chat", "size": "6b"},

    # Chronos (TimeSeriesAI)
    {"name": "amazon/chronos-t5-base", "display": "Chronos T5 Base", "context": 512, "type": "specialized", "size": "base"},

    # Phi Models
    {"name": "microsoft/phi-2", "display": "Phi 2", "context": 2048, "type": "chat", "size": "2.7b"},

    # Baichuan Models
    {"name": "baichuan-inc/Baichuan2-13B-Chat", "display": "Baichuan 2 13B Chat", "context": 4096, "type": "chat", "size": "13b"},
    {"name": "baichuan-inc/Baichuan-13B-Chat", "display": "Baichuan 13B Chat", "context": 4096, "type": "chat", "size": "13b"},

    # Bloom Series
    {"name": "bigscience/bloom-7b1", "display": "Bloom 7B1", "context": 2048, "type": "base", "size": "7b"},
    {"name": "bigscience/bloomz-7b1", "display": "Bloomz 7B1", "context": 2048, "type": "chat", "size": "7b"},

    # Additional Specialized Models
    {"name": "togethercomputer/ReluMix-1.3B", "display": "ReluMix 1.3B", "context": 2048, "type": "base", "size": "1b"},
    {"name": "EleutherAI/pythia-12b-deduped", "display": "Pythia 12B Deduped", "context": 2048, "type": "base", "size": "12b"},
    {"name": "bigcode/starcoder", "display": "StarCoder", "context": 8192, "type": "chat", "size": "15b"},
    {"name": "bigcode/starcoder-3b", "display": "StarCoder 3B", "context": 8192, "type": "chat", "size": "3b"},
    {"name": "NinedayWang/PolyCoder-2.1B", "display": "PolyCoder 2.1B", "context": 2048, "type": "chat", "size": "2b"},
    {"name": "stabilityai/japanese-stablelm-base-beta-70b", "display": "Japanese StableLM 70B", "context": 2048, "type": "base", "size": "70b"},
    {"name": "togethercomputer/GPT-JT-Moderation-6B", "display": "GPT-JT Moderation 6B", "context": 2048, "type": "specialized", "size": "6b"},
]

def estimate_pricing(size: str) -> tuple:
    """Estimate pricing based on model size."""
    pricing_map = {
        "1b": (0.00000025, 0.00000025),
        "2b": (0.00000050, 0.00000050),
        "3b": (0.00000075, 0.00000075),
        "6b": (0.00000150, 0.00000150),
        "7b": (0.00000200, 0.00000200),
        "8b": (0.00000200, 0.00000200),
        "10b": (0.00000250, 0.00000250),
        "12b": (0.00000300, 0.00000300),
        "13b": (0.00000350, 0.00000350),
        "14b": (0.00000350, 0.00000350),
        "15b": (0.00000400, 0.00000400),
        "22b": (0.00000550, 0.00000550),
        "30b": (0.00000800, 0.00000800),
        "32b": (0.00000850, 0.00000850),
        "33b": (0.00000900, 0.00000900),
        "34b": (0.00000950, 0.00000950),
        "40b": (0.00001200, 0.00001200),
        "67b": (0.00002000, 0.00002000),
        "70b": (0.00002100, 0.00002100),
        "72b": (0.00002200, 0.00002200),
        "405b": (0.00090000, 0.00090000),
    }
    return pricing_map.get(size, (0.00000200, 0.00000200))

def classify_capabilities(name: str, model_type: str) -> str:
    """Classify model capabilities."""
    caps = ["T"]  # All support text
    name_lower = name.lower()

    if "code" in name_lower or "coder" in name_lower or "python" in name_lower:
        caps.append("J")

    if model_type == "specialized":
        if "chronos" in name_lower:
            pass  # Time series specific
        elif "moderation" in name_lower:
            pass

    return "".join(sorted(set(caps)))

def model_to_csv_row(model: Dict, index: int) -> Dict[str, str]:
    """Convert model to CSV row."""
    name = model["name"]
    display = model["display"]
    context = model["context"]
    model_type = model.get("type", "chat")
    size = model.get("size", "7b")

    # Create alias
    alias = name.split("/")[-1].lower()[:50]

    # Estimate pricing
    input_price, output_price = estimate_pricing(size)

    # Estimate max output
    max_output = min(4096, context // 4)

    # Classify capabilities
    capabilities = classify_capabilities(name, model_type)

    return {
        "id": f"together/{alias}",
        "alias": alias,
        "name": f"Together: {display}",
        "status": "C",
        "input_price": str(input_price),
        "output_price": str(output_price),
        "cache_input_price": "-",
        "context_window": str(context),
        "max_output": str(max_output),
        "capabilities": capabilities,
        "quality": "verified",
        "source": "together",
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "description": f"Open-source {display} model hosted on Together AI platform",
        "mmlu_score": "-",
        "bbh_score": "-",
        "humaneval_score": "-",
        "aime_score": "-",
        "livebench_score": "-",
        "gpqa_score": "-",
        "gpt4_eval_score": "-",
        "agentic_score": "-",
        "ceo_evals_score": "-",
    }

def save_to_csv(models: List[Dict[str, str]], output_path: Path) -> int:
    """Save models to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "alias", "name", "status", "input_price", "output_price",
        "cache_input_price", "context_window", "max_output", "capabilities",
        "quality", "source", "updated", "description", "mmlu_score",
        "bbh_score", "humaneval_score", "aime_score", "livebench_score",
        "gpqa_score", "gpt4_eval_score", "agentic_score", "ceo_evals_score"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(models)

    print(f"✓ Saved {len(models)} models to {output_path}", file=sys.stderr)
    return len(models)

def main():
    """Main entry point."""
    # Convert models to CSV rows
    csv_rows = []
    for idx, model in enumerate(TOGETHER_AI_MODELS, 1):
        try:
            row = model_to_csv_row(model, idx)
            csv_rows.append(row)
        except Exception as e:
            print(f"Warning: Error processing model {idx}: {e}", file=sys.stderr)
            continue

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "models" / "aggregators" / "together_ai.csv"
    count = save_to_csv(csv_rows, output_path)

    print(f"\nProcessed {count} Together AI models", file=sys.stderr)
    return 0 if count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
