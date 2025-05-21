import os
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import nltk
from rouge_score import rouge_scorer
import numpy as np

nltk.download('punkt')

DATA_DIR = '/workspace/AAAproj/data/main_dataset'
IMG_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'meta_info.csv')
MODEL_ID = 'tuman/vit-rugpt2-image-captioning'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Скачиваем модель и процессор локально через snapshot_download
MODEL_LOCAL_PATH = './hf_models/vit-rugpt2-image-captioning'
if not os.path.exists(MODEL_LOCAL_PATH):
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL_PATH)


def mask_sizes(text: str) -> str:
    '''
    Маскирует размеры и числовые параметры в тексте

    Parameters
    ----------
    text : str
        Исходный текст

    Returns
    -------
    str
        Текст с маскированными размерами
    '''
    if not isinstance(text, str):
        return ''
    # заменяем размеры и числовые параметры
    text = re.sub(r'\d+([.,]\d+)?\s*(см|mm|мм|м|г|гр|kg|кг|размер|длина|обхват|объем|ширина|рост|по стельке)', '__ \\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\d+([.,]\d+)?\s*(см|mm|мм|м)\)', '(__ \\2)', text)
    return text

def attrs_to_text(attrs_str: str) -> str:
    '''
    Преобразует строку с атрибутами в читаемый текст

    Parameters
    ----------
    attrs_str : str
        Строка с атрибутами (dict)

    Returns
    -------
    str
        Текстовое представление атрибутов
    '''
    try:
        attrs = eval(attrs_str)
        return '. '.join([f'{k}: {v}' for k, v in attrs.items()])
    except Exception:
        return attrs_str if isinstance(attrs_str, str) else ''

def resize_with_padding(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    '''
    Изменяет размер изображения с сохранением пропорций и добавляет белые поля

    Parameters
    ----------
    img : Image.Image
        Исходное изображение
    target_size : tuple[int, int]
        Желаемый размер (ширина, высота)

    Returns
    -------
    Image.Image
        Изображение нужного размера с паддингом
    '''
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def make_collage(image_paths: list[str], grid_size=(2, 5), image_size=(224, 224)) -> Image.Image:
    '''
    Создаёт коллаж из изображений с паддингом для сохранения пропорций

    Parameters
    ----------
    image_paths : list[str]
        Пути к изображениям
    grid_size : tuple
        Размер сетки (строки, столбцы)
    image_size : tuple
        Размер одного изображения

    Returns
    -------
    Image.Image
        Коллаж из изображений
    '''
    n_images = len(image_paths)
    grid_rows, grid_cols = grid_size
    total_slots = grid_rows * grid_cols

    images = []
    for i in range(total_slots):
        if i < n_images:
            try:
                img = Image.open(image_paths[i]).convert('RGB')
                img = resize_with_padding(img, image_size)
            except Exception:
                img = Image.new('RGB', image_size, (255, 255, 255))
        else:
            img = Image.new('RGB', image_size, (255, 255, 255))
        images.append(img)

    collage_w = image_size[0] * grid_cols
    collage_h = image_size[1] * grid_rows
    collage = Image.new('RGB', (collage_w, collage_h), (255, 255, 255))
    for idx, img in enumerate(images):
        row, col = idx // grid_cols, idx % grid_cols
        collage.paste(img, (col * image_size[0], row * image_size[1]))
    return collage

class CaptionCollageDataset(Dataset):
    '''
    Датасет для генерации описаний к коллажам изображений

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    processor : AutoProcessor
        Процессор для обработки изображений и текста
    img_dir : str
        Путь к папке с изображениями
    max_images : int
        Максимум изображений в коллаже
    '''
    def __init__(self, df: pd.DataFrame, processor, img_dir: str, max_images=10):
        self.df = df
        self.processor = processor
        self.img_dir = img_dir
        self.max_images = max_images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # изображения
        image_ids = eval(row['images'])
        image_paths = [os.path.join(self.img_dir, f'{img_id}.jpg') for img_id in image_ids[:self.max_images]]
        collage = make_collage(image_paths, grid_size=(2, 5), image_size=(224, 224))
        # текст
        prompt = f"{row['title']}. {attrs_to_text(row['attrs'])}"
        # маскируем описание
        target = mask_sizes(row['description']) if pd.notna(row['description']) else ''
        if not target and pd.notna(row['other']):
            target = mask_sizes(row['other'])
        # вход для модели
        model_inputs = self.processor(images=collage, text=prompt, return_tensors='pt')
        labels = self.processor.tokenizer(target, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        model_inputs['labels'] = labels['input_ids'].squeeze(0)
        return model_inputs

print('reading csv...')
df = pd.read_csv(CSV_PATH)
df = df[df['images'].notnull() & df['title'].notnull() & df['attrs'].notnull()]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'total rows after filtering: {len(df)}')

# train/val/test split с RANDOM_STATE=42
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.05, random_state=42)

print(f'train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')

processor = AutoProcessor.from_pretrained(MODEL_LOCAL_PATH)
train_dataset = CaptionCollageDataset(train_df, processor, IMG_DIR, max_images=10)
val_dataset = CaptionCollageDataset(val_df, processor, IMG_DIR, max_images=10)
test_dataset = CaptionCollageDataset(test_df, processor, IMG_DIR, max_images=10)

model = AutoModelForVision2Seq.from_pretrained(MODEL_LOCAL_PATH).to(DEVICE)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=2,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    report_to='none',
    fp16=True,
    gradient_accumulation_steps=2,
    save_total_limit=2,
    remove_unused_columns=False,
)

def compute_metrics(pred) -> dict:
    '''
    Вычисляет BLEU и ROUGE-L для предсказаний

    Parameters
    ----------
    pred : PredictionOutput
        Результаты предсказания Trainer

    Returns
    -------
    dict
        Метрики BLEU и ROUGE-L
    '''
    pred_str = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    bleu = np.mean([
        nltk.translate.bleu_score.sentence_bleu([ref.split()], hyp.split())
        if len(ref.split()) > 0 else 0 for ref, hyp in zip(label_str, pred_str)
    ])
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = np.mean([
        scorer.score(ref, hyp)['rougeL'].fmeasure if ref and hyp else 0
        for ref, hyp in zip(label_str, pred_str)
    ])
    return {'bleu': bleu, 'rougeL': rouge}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print('starting training...')
trainer.train()
print('saving checkpoint...')
trainer.save_model(CHECKPOINT_DIR)

print('evaluating final metrics on validation:')
val_metrics = trainer.evaluate()
print(val_metrics)

print('evaluating final metrics on test:')
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(test_metrics)


def infer(images: list[str], title: str, attrs: str, model: AutoModelForVision2Seq, processor, device=DEVICE) -> str:
    '''
    Генерирует описание для набора изображений и атрибутов

    Parameters
    ----------
    images : list[str]
        Пути к изображениям
    title : str
        Заголовок товара
    attrs : str
        Атрибуты товара (dict или строка)
    model : AutoModelForVision2Seq
        Модель для генерации
    processor : AutoProcessor
        Процессор для подготовки входа
    device : str
        Устройство для инференса

    Returns
    -------
    str
        Сгенерированное описание
    '''
    collage = make_collage(images, grid_size=(2, 5), image_size=(224, 224))
    prompt = f'{title}. {attrs_to_text(attrs)}'
    inputs = processor(images=collage, text=prompt, return_tensors='pt').to(device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.95, num_return_sequences=1)
    return processor.tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == '__main__':
    if len(val_df) > 0:
        item = val_df.iloc[0]
        image_ids = eval(item['images'])[:10]
        image_paths = [os.path.join(IMG_DIR, f'{img_id}.jpg') for img_id in image_ids]
        print('generated description example:')
        print(infer(image_paths, item['title'], item['attrs'], model, processor))
