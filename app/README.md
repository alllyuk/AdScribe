# Инструкция по запуску

1. Установите зависимости:

```bash
pip install fastapi uvicorn jinja2 requests
```

2. Запустите сервер с автообновлением:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Откройте в браузере: http://localhost:8000


5. Для добавления тест-кейсов создайте папки внутри `test_cases`, положите туда картинку (image.jpg/png) и файл признаков (desc.txt).
