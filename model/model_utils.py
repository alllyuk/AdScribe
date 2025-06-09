import yaml
import os
from loguru import logger
from model.models import ViTRuGPT2CaptionModel, BLIP2LLMModel
from model.qwen_finetuned import QwenVisionModel


def get_caption_model(config: dict):
    model_type = config["model"]["type"]
    if model_type == "vit_rugpt2":
        model_name = config["model"]["vit_rugpt2"]["model_name"]
        return ViTRuGPT2CaptionModel(model_name)
    elif model_type == "blip2_llm":
        blip_id = config["model"]["blip2_llm"]["blip_model_id"]
        llm_id = config["model"]["blip2_llm"]["llm_model_id"]
        lora_path = config["model"]["blip2_llm"].get("lora_path")
        return BLIP2LLMModel(blip_id, llm_id, lora_path)
    elif model_type == "qwen_vision":
        base_model_name = config["model"]["qwen_vision"].get(
            "base_model_name", "unsloth/Qwen2.5-VL-7B-Instruct"
        )
        lora_adapter_name = config["model"]["qwen_vision"].get(
            "lora_adapter_name", "moxeeeem/aaa_proj2"
        )
        classifier_model_path = config["model"]["qwen_vision"].get(
            "classifier_model_path", "/workspace/AAA_project/qwen_usage/model.pt"
        )
        load_in_4bit = config["model"]["qwen_vision"].get("load_in_4bit", True)
        return QwenVisionModel(
            base_model_name=base_model_name,
            lora_adapter_name=lora_adapter_name,
            classifier_model_path=classifier_model_path,
            load_in_4bit=load_in_4bit,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_config(config_path="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
