from abc import ABC, abstractmethod
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
)
import torch
from PIL import Image
import io
from peft import PeftModel
import yaml
import os
from huggingface_hub import snapshot_download
from loguru import logger


class BaseCaptionModel(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes, features: str = None) -> str:
        pass


class ViTRuGPT2CaptionModel(BaseCaptionModel):
    def __init__(self, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            snapshot_download(repo_id=model_name, local_dir=model_dir)
            logger.debug(f"Donwloaded model snapshot to {model_dir}")

        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gen_kwargs = {
            "max_length": 64,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 1.0,
        }

    def predict(self, image_bytes: bytes, features: str = None) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = self.feature_extractor(
            images=[image], return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class BLIP2LLMModel(BaseCaptionModel):
    def __init__(self, blip_model_id, llm_model_id, lora_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_processor = Blip2Processor.from_pretrained(blip_model_id)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_id, torch_dtype=torch.float16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id, torch_dtype=torch.float16, load_in_8bit=True
        ).to(self.device)
        if lora_path:
            self.llm = PeftModel.from_pretrained(self.llm, lora_path)

    def predict(self, image_bytes: bytes, features: str = None) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        visual_prompt = "Describe the image briefly"
        inputs = self.blip_processor(image, text=visual_prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output_ids = self.blip_model.generate(**inputs, max_new_tokens=32)
        visual_desc = self.blip_processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        full_prompt = f"""
            На изображении показан предмет: {visual_desc}.
            Характеристики:
            {features.strip() if features else ''}
            Сформулируй привлекательное и лаконичное описание для онлайн-объявления:
            """

        llm_inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.llm.generate(
                **llm_inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return (
            self.tokenizer.decode(output[0], skip_special_tokens=True)
            .split("объявления:")[-1]
            .strip()
        )


def get_caption_model(config: dict):
    model_type = config["model"]["type"]
    if model_type == "vit_rugpt2":
        model_dir = config["model"]["vit_rugpt2"]["model_dir"]
        model_name = config["model"]["vit_rugpt2"]["model_name"]
        return ViTRuGPT2CaptionModel(model_dir, model_name)
    elif model_type == "blip2_llm":
        blip_id = config["model"]["blip2_llm"]["blip_model_id"]
        llm_id = config["model"]["blip2_llm"]["llm_model_id"]
        lora_path = config["model"]["blip2_llm"].get("lora_path")
        return BLIP2LLMModel(blip_id, llm_id, lora_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_config(config_path="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- OLD CODE ---
# def parse_features(features_text):
#     # TODO: реализовать парсинг признаков для мультимодальных моделей
#     return features_text
