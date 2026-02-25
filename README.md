pip install -r requirements.txt
uvicorn api.index:app --reload --env-file .env
