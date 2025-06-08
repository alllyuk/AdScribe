from abc import ABC, abstractmethod
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
)
import torch
from PIL import Image
import io
from peft import PeftModel
import yaml
import os
from huggingface_hub import snapshot_download, login
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


class BaseCaptionModel(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes, features: str = None) -> str:
        pass


class ViTRuGPT2CaptionModel(BaseCaptionModel):
    def __init__(self, model_dir, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            snapshot_download(repo_id=model_name, local_dir=model_dir)
            logger.debug(f"Downloaded model snapshot to {model_dir}")

        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
    def __init__(self, model_dir, blip_model_id, llm_model_id, lora_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        blip_dir = os.path.join(model_dir, "blip")
        llm_dir = os.path.join(model_dir, "llm")

        if not os.path.exists(os.path.join(blip_dir, "pytorch_model.bin")):
            snapshot_download(
                repo_id=blip_model_id, local_dir=blip_dir, local_dir_use_symlinks=False
            )
            logger.debug(f"Downloaded BLIP model to {blip_dir}")

        if not os.path.exists(os.path.join(llm_dir, "pytorch_model.bin")):
            snapshot_download(
                repo_id=llm_model_id, local_dir=llm_dir, local_dir_use_symlinks=False
            )
            logger.debug(f"Downloaded LLM model to {llm_dir}")

        self.blip_processor = AutoProcessor.from_pretrained(blip_dir, use_fast=True)
        self.blip_model = AutoModelForImageTextToText.from_pretrained(
            blip_dir, torch_dtype=torch.float16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, use_fast=True)
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_dir, torch_dtype=torch.float16, load_in_8bit=True
            )
        except Exception as e:
            logger.warning(
                f"Failed to load in 8bit: {e}. Falling back to full precision."
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_dir, torch_dtype=torch.float16
            ).to(self.device)

        if lora_path:
            self.llm = PeftModel.from_pretrained(self.llm, lora_path).to(self.device)

    def predict(self, image_bytes: bytes, features: str = None) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # visual_prompt = "Describe the image briefly"
        inputs = self.blip_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.blip_model.generate(**inputs, max_new_tokens=50)

        visual_desc = self.blip_processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        logger.debug(f"Image description from BLIP: {visual_desc}")

        full_prompt = f"""
            На изображении показан предмет одежды или другой предмет / товар (информация дана на английском языке, нужно перевести ее и использовать): {visual_desc}.
            Характеристики этого предмета / товара: {features.strip() if features else 'отсутствуют'}.
            Сформулируй привлекательное и лаконичное описание для онлайн-объявления на основе полученной информации.
            В спорной ситуации между информацией на английском и характеристиками предмета выбирай информацию из характеристик.
            Объявление должно быть на русском языке.
            Если информация не содержит характеристик объявления, запиши разумные характеристики для этого предмета,
            заменяя точные значения характеристик плейсхолдерами нижними подчеркиваниями, например: цвет ____, размер ____, бренд ____.
            Бери информацию о бренде только из характеристик товара, не бери ее из информации на английском языке.

            Пример:
            Описание предмета на английском: blue checkered men's jacket
            Характеристики предмета: цвет: синий, размер: 50, состояние: почти новое, бренд: Emporio Armani.

            Итоговое описание (как ты должен делать):
            Продаю стильный мужской пиджак от знаменитого бренда Emporio Armani. Цвет — синий с клетчатым узором, размер 50 (европейская маркировка). Состояние почти новое — надевался всего несколько раз, бережно хранился.
            Идеально подойдет как для деловых встреч, так и для повседневного образа. Качественный пошив, премиальные материалы и безупречный итальянский стиль.

            Не форматируй описание никакими спецсимволами. Также не пиши в начале описания ничего типа "твой результат", "итоговое описание" или другого.
            Верни сразу конечный вид описания без дополнительных указаний и пояснений.
            """

        llm_inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        output_ids = self.llm.generate(
            **llm_inputs,
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = llm_inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][input_length:]

        result = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return result


def get_caption_model(config: dict):
    model_type = config["model"]["type"]
    if model_type == "vit_rugpt2":
        model_dir = config["model"]["vit_rugpt2"]["model_dir"]
        model_name = config["model"]["vit_rugpt2"]["model_name"]
        return ViTRuGPT2CaptionModel(model_dir, model_name)
    elif model_type == "blip2_llm":
        model_dir = config["model"]["blip2_llm"]["model_dir"]
        blip_id = config["model"]["blip2_llm"]["blip_model_id"]
        llm_id = config["model"]["blip2_llm"]["llm_model_id"]
        lora_path = config["model"]["blip2_llm"].get("lora_path")
        return BLIP2LLMModel(model_dir, blip_id, llm_id, lora_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_config(config_path="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
