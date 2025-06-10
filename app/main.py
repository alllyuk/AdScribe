import os
import time
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List
from model.model_utils import (
    get_caption_model,
    load_config,
)
from loguru import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


config = load_config()
caption_model = get_caption_model(config)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    test_cases = []
    if os.path.exists(TEST_CASES_DIR):
        for d in os.listdir(TEST_CASES_DIR):
            if os.path.isdir(os.path.join(TEST_CASES_DIR, d)):
                test_cases.append(d)
        logger.debug(f"Initialized test cases from {TEST_CASES_DIR}")

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
            logger.debug("Generate from test case")
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
            logger.debug("Generate from user input")
            if images:
                for img in images:
                    image_datas.append(await img.read())
            if features_text.strip():
                parsed_features = features_text.strip()
        if not image_datas:
            logger.debug("No one images uploaded")
            raise Exception("Нужно загрузить хотя бы одну картинку!")

        logger.debug("Data uploaded, start generation")

        preds = [caption_model.predict(img, parsed_features) for img in image_datas]

        logger.debug("Descriptions are ready")

        metrics["len_chars"] = [len(t) for t in preds]
        metrics["len_words"] = [len(t.split()) for t in preds]
        result = {"generated_texts": preds}
    except Exception as e:
        error = str(e)
    elapsed = round(time.time() - start_time, 2)
    return JSONResponse(
        {"result": result, "error": error, "metrics": metrics, "elapsed": elapsed}
    )
