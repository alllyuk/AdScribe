import os
from huggingface_hub import snapshot_download


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


class BaseCaptionModel(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes, features: str = None) -> str:
        pass


class ViTRuGPT2CaptionModel(BaseCaptionModel):
    def __init__(self, model_dir):
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

        full_prompt = f"""На изображении показан предмет: {visual_desc}.
Характеристики:
{features.strip() if features else ''}
Сформулируй привлекательное и лаконичное описание для онлайн-объявления:"""

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


def get_caption_model(model_type: str):
    if model_type == "vit-rugpt2":
        return ViTRuGPT2CaptionModel("model_snapshot")
    elif model_type == "blip2-llm":
        return BLIP2LLMModel(
            blip_model_id="Salesforce/blip2-flan-t5-xl",
            llm_model_id="mistralai/Mistral-7B-Instruct-v0.2",
            lora_path=None,  # если используешь LoRA
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# --- Заглушка для будущей мультимодальности ---
# --- Заглушка для будущей мультимодальности ---
# --- Заглушка для будущей мультимодальности ---
MODEL_NAME = "tuman/vit-rugpt2-image-captioning"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_snapshot")

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
        resume_download=False,
    )

model = VisionEncoderDecoderModel.from_pretrained(LOCAL_MODEL_DIR)
feature_extractor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
# REPLACE IT TO MAKE MODEL WORKS WITHOUT BEAN SEARCH
gen_kwargs = {
    "max_length": 64,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 1.0,
}


def predict_caption_single_image(image_bytes):
    i_image = Image.open(io.BytesIO(image_bytes))
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    pixel_values = feature_extractor(
        images=[i_image], return_tensors="pt"
    ).pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return pred


# --- Заглушки для мультимодальных функций ---
def predict_multimodal(*args, **kwargs):
    # TODO: реализовать мультимодальную генерацию
    return "[Мультимодальная генерация не реализована]"


def parse_features(features_text):
    # TODO: реализовать парсинг признаков для мультимодальных моделей
    return features_text
