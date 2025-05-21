import os
import time
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List
from huggingface_hub import snapshot_download
from model_utils import predict_caption_single_image, predict_multimodal, parse_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
MODEL_NAME = "tuman/vit-rugpt2-image-captioning"
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "model_snapshot")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Скачиваем snapshot модели в локальную папку (один раз, затем используем локально)
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        snapshot_download(repo_id=MODEL_NAME, local_dir=LOCAL_MODEL_DIR)

    test_cases = []
    if os.path.exists(TEST_CASES_DIR):
        for d in os.listdir(TEST_CASES_DIR):
            if os.path.isdir(os.path.join(TEST_CASES_DIR, d)):
                test_cases.append(d)
    return templates.TemplateResponse(
        "index.html", {"request": request, "test_cases": test_cases}
    )


@app.post("/generate")
async def generate(
    request: Request,
    images: List[UploadFile] = File(None),
    features: Optional[UploadFile] = File(None),
    features_text: Optional[str] = Form(None),
    test_case: Optional[str] = Form(None),
):
    start_time = time.time()
    error = None
    result = None
    metrics = {}
    try:
        image_datas = []
        parsed_features = None
        if test_case:
            case_path = os.path.join(TEST_CASES_DIR, test_case)
            for fname in os.listdir(case_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    with open(os.path.join(case_path, fname), "rb") as f:
                        image_datas.append(f.read())
                elif fname.lower().endswith(".txt"):
                    with open(
                        os.path.join(case_path, fname), "r", encoding="utf-8"
                    ) as f:
                        parsed_features = f.read()
        else:
            if images:
                for img in images:
                    image_datas.append(await img.read())
            if features:
                parsed_features = (await features.read()).decode("utf-8")
            if features_text and features_text.strip():
                parsed_features = features_text.strip()
        if not image_datas:
            raise Exception("Нужно загрузить хотя бы одну картинку!")
        # Только картинка -> текст (одна картинка — один текст)
        preds = [predict_caption_single_image(img) for img in image_datas]
        # Заглушка: если будут признаки — использовать predict_multimodal
        # preds = [predict_multimodal(img, parsed_features) for img in image_datas]
        metrics["len_chars"] = [len(t) for t in preds]
        metrics["len_words"] = [len(t.split()) for t in preds]
        result = {"generated_texts": preds}
    except Exception as e:
        error = str(e)
    elapsed = round(time.time() - start_time, 2)
    return JSONResponse(
        {"result": result, "error": error, "metrics": metrics, "elapsed": elapsed}
    )
