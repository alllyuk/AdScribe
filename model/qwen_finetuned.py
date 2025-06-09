from abc import ABC, abstractmethod
from unsloth import FastVisionModel
import torch
import gc
import numpy as np
import re
from PIL import Image
import ast
import timm
import torch.nn as nn
import torchvision.transforms as transforms
import os
import emoji
import string
import pandas as pd
from loguru import logger


class BaseCaptionModel(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes, features: str = None) -> str:
        pass


class ProductCountClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50.a1_in1k", num_classes=2, p_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )

        self.bn1 = nn.BatchNorm1d(self.backbone.num_features)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.fc1 = nn.Linear(self.backbone.num_features, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=p_dropout / 2)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=p_dropout / 4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        return self.fc3(x)


class QwenVisionModel(BaseCaptionModel):
    def __init__(
        self,
        base_model_name="unsloth/Qwen2.5-VL-7B-Instruct",
        lora_adapter_name="moxeeeem/aaa_proj2",
        classifier_model_path="/workspace/AAA_project/qwen_usage/model.pt",
        load_in_4bit=True,
    ):
        """
        Инициализация модели Qwen2.5-VL с LoRA адаптерами

        Args:
            base_model_name: Название базовой модели
            lora_adapter_name: Название LoRA адаптера
            classifier_model_path: Путь к модели классификации количества товаров
            load_in_4bit: Загружать ли модель в 4-битном режиме
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.lora_adapter_name = lora_adapter_name
        self.classifier_model_path = classifier_model_path
        self.load_in_4bit = load_in_4bit

        self.class_names = ["много_товаров", "один_товар"]
        self.category_mapping = {
            "(Ж) Обувь": "Женская обувь",
            "(М) Обувь": "Мужская обувь",
            "(Ж) Комбинезоны": "Женские комбинезоны",
            "(М) Комбинезоны": "Мужские комбинезоны",
            "(Ж) Верхняя одежда": "Женская верхняя одежда",
            "(М) Верхняя одежда": "Мужская верхняя одежда",
            "(Ж) Платья и юбки": "Женские платья и юбки",
            "(М) Пиджаки и костюмы": "Мужские пиджаки и костюмы",
            "(Ж) Пиджаки и костюмы": "Женские пиджаки и костюмы",
            "(М) Брюки": "Мужские брюки",
            "(Ж) Брюки": "Женские брюки",
            "(М) Свитеры и толстовки": "Мужские свитеры и толстовки",
            "(Ж) Свитеры и толстовки": "Женские свитеры и толстовки",
            "(Ж) Блузки и рубашки": "Женские блузки и рубашки",
            "(М) Рубашки": "Мужские рубашки",
            "(Ж) Футболки и топы": "Женские футболки и топы",
            "(М) Футболки и майки": "Мужские футболки и майки",
        }

        self._load_models()

    def _load_models(self):
        """Загрузка всех моделей"""
        logger.info("Загружаем модель классификации...")
        self._load_classifier()

        logger.info("Загружаем vision модель...")
        self._load_vision_model()

    def _load_classifier(self):
        """Загрузка модели классификации количества товаров"""
        self.classifier_model = ProductCountClassifier().to(self.device)
        self.classifier_model.load_state_dict(
            torch.load(self.classifier_model_path, map_location=self.device)
        )
        self.classifier_model.eval()
        logger.info("Модель классификации загружена")

    def _load_vision_model(self):
        """Загрузка vision модели с LoRA адаптерами"""
        # Очистка памяти
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Загружаем базовую модель
        self.vision_model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.base_model_name,
            load_in_4bit=self.load_in_4bit,
        )

        # Загружаем LoRA адаптеры
        self.vision_model = FastVisionModel.get_peft_model(
            self.vision_model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Загружаем веса LoRA адаптеров
        self.vision_model.load_adapter(self.lora_adapter_name, adapter_name="default")

        FastVisionModel.for_inference(self.vision_model)
        logger.info("Vision модель загружена")

    def get_transform(self):
        """Получение трансформаций для изображений"""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_single_image(self, img_path, transform):
        """Предсказание для одного изображения"""
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.classifier_model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            return {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        except Exception as e:
            logger.error(f"Ошибка при обработке {img_path}: {e}")
            return None

    def predict_image_set(self, img_paths, transform):
        """Предсказание для набора изображений"""
        all_probs = []

        for img_path in img_paths:
            pred = self.predict_single_image(img_path, transform)
            if pred is not None:
                all_probs.append([pred[name] for name in self.class_names])

        if not all_probs:
            return {name: 0.0 for name in self.class_names}

        avg_probs = np.mean(all_probs, axis=0)
        return {name: float(avg_probs[i]) for i, name in enumerate(self.class_names)}

    def remove_emojis(self, text):
        """Удаление эмодзи из текста"""
        return emoji.replace_emoji(text, replace="")

    def mask_text(self, text: str) -> str:
        """Маскировка персональных данных в тексте"""
        if not text or pd.isna(text):
            return text

        text = str(text)

        text = re.sub(r"\d+[.,]\d+", "__", text)  # 3.5, 5,5
        text = re.sub(r"\d+\(\d+\)", "__", text)  # 55(53)
        text = re.sub(r"\+?\d[\d\s\-\(\)]{6,}", "__", text)  # телефоны
        text = re.sub(r"\b\d+\b", "__", text)  # любые цифры

        text = re.sub(r"https?://\S+", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"www\.\S+", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"\S+\.(ru|com|рф|org|net)\S*", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"\S+@\S+\.\S+", "__", text, flags=re.IGNORECASE)  # email

        text = re.sub(r"г\.\s*\S+.*?(?=\.|$|\n)", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"город\s+\S+.*?(?=\.|$|\n)", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"ул\.\s*\S+.*?(?=\.|$|\n)", "__", text, flags=re.IGNORECASE)
        text = re.sub(r"улица\s+\S+.*?(?=\.|$|\n)", "__", text, flags=re.IGNORECASE)
        text = re.sub(
            r"адрес\s*:?\s*\S+.*?(?=\.|$|\n)", "__", text, flags=re.IGNORECASE
        )

        # множественные __ в один
        text = re.sub(r"\[HIDDEN\](\s*\[HIDDEN\])+", "__", text)

        # убираем пробелы перед запятыми
        text = re.sub(r"\s+,", ",", text)

        # расстановка пробелов вокруг __
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([^\s([.,;:!?-])[HIDDEN]", r"\1 ", text)
        text = re.sub(r"[HIDDEN]([^\s)].,;:!?-])", r" \1", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_and_create_query(self, title, attrs, images, category):
        """Предобработка данных и создание запроса для модели"""
        # 1. Предобработка категории
        processed_category = self.category_mapping.get(category, category)

        # 2. Предобработка title
        processed_title = str(title)
        processed_title = re.sub(r"[\n\t]", " ", processed_title).strip()
        processed_title = self.remove_emojis(processed_title)
        processed_title = re.sub(r"\s{2,}", " ", processed_title).strip()
        processed_title = re.sub(
            r"([" + re.escape(string.punctuation) + r"])\1+", r"\1", processed_title
        ).strip()
        processed_title = self.mask_text(processed_title)

        # 3. Предобработка attrs
        if isinstance(attrs, str):
            try:
                attrs_dict = ast.literal_eval(attrs)
            except:
                attrs_dict = {}
        else:
            attrs_dict = attrs if isinstance(attrs, dict) else {}

        processed_attrs = {}
        for k, v in attrs_dict.items():
            masked_v = self.mask_text(str(v))
            processed_attrs[k] = masked_v

        attrs_str = ", ".join([f"{k}: {v}" for k, v in processed_attrs.items()])

        # 4. Обработка путей к изображениям
        processed_images = images if isinstance(images, list) else [images]

        # 5. Классификация количества товаров
        transform = self.get_transform()

        if len(processed_images) == 1:
            pred = self.predict_single_image(processed_images[0], transform)
            if pred is not None:
                multiitem_proba = pred["много_товаров"]
            else:
                multiitem_proba = 0.0
        else:
            pred = self.predict_image_set(processed_images, transform)
            multiitem_proba = pred["много_товаров"]

        # 6. Создание запроса
        query = f"Создай краткое и привлекательное описание для объявления о продаже товара или ассортимента товаров. Оно будет размещено на e-commerce платформе. Название товара: {processed_title}. Атрибуты товара: {attrs_str}. Категория товара: {processed_category}. Вероятность, что в объявлении много товаров: {multiitem_proba:.2f}. Опиши товар естественно, без списков и структуры, на русском языке. Обязательно упомянуть характеристи товара, которые ты узнал."

        return query

    def resize_image(self, image_path, max_size=512):
        """Изменение размера изображения"""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_path}: {e}")
            return None

    def predict(self, image_bytes: bytes, features: str = None) -> str:
        """
        Основной метод для предсказания описания товара

        Args:
            image_bytes: Байты изображения
            features: Дополнительные характеристики в формате строки

        Returns:
            str: Сгенерированное описание товара
        """
        try:
            # Сохраняем временное изображение
            temp_image_path = "/tmp/temp_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_bytes)

            # Подготавливаем изображение
            resized_img = self.resize_image(temp_image_path, max_size=512)
            if resized_img is None:
                return "Ошибка при обработке изображения"

            # Создаем запрос (используем пустые значения для недостающих параметров)
            query = self.preprocess_and_create_query(
                title="",
                attrs=features if features else "",
                images=[temp_image_path],
                category="",
            )

            # Подготавливаем сообщение для модели
            content = [
                {"type": "image", "image": resized_img},
                {"type": "text", "text": query},
            ]

            messages = [{"role": "user", "content": content}]

            # Генерируем предсказание
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                [resized_img],
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                outputs = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    min_new_tokens=10,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            return prediction

        except Exception as e:
            logger.error(f"Ошибка при генерации описания: {e}")
            return f"Ошибка при генерации описания: {str(e)}"

    def predict_from_test_case(self, test_case_path):
        """
        Предсказание из тест-кейса

        Args:
            test_case_path: Путь к папке с тест-кейсом

        Returns:
            tuple: (предсказание, входной текст)
        """
        try:
            # 1. Загружаем CSV файл
            csv_path = os.path.join(test_case_path, "data.csv")
            if not os.path.exists(csv_path):
                logger.error(f"CSV файл не найден: {csv_path}")
                return "", ""

            df = pd.read_csv(csv_path)
            if len(df) == 0:
                logger.error("CSV файл пуст")
                return "", ""

            row = df.iloc[0]

            # 2. Подготавливаем пути к изображениям
            image_files = [f for f in os.listdir(test_case_path) if f.endswith(".jpg")]
            image_paths = [
                os.path.join(test_case_path, img_file) for img_file in image_files
            ]

            if not image_paths:
                logger.error(f"Изображения не найдены в папке: {test_case_path}")
                return "", ""

            # 3. Создаем запрос
            query = self.preprocess_and_create_query(
                row["title"],
                row["attrs"],
                image_paths,
                row["category"],
            )

            # 4. Подготавливаем изображения
            images = []
            for img_path in image_paths:
                resized_img = self.resize_image(img_path, max_size=512)
                if resized_img is not None:
                    images.append(resized_img)

            if not images:
                logger.error("Нет валидных изображений для обработки")
                return "", ""

            # 5. Подготавливаем сообщение
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": query})

            messages = [{"role": "user", "content": content}]

            # 6. Генерируем предсказание
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                outputs = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    min_new_tokens=10,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            return prediction, input_text

        except Exception as e:
            logger.error(f"Ошибка при обработке тест-кейса {test_case_path}: {e}")
            return "", ""
