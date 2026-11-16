# CPU image for the TRACE-X dashboard / API.
# Processed data and checkpoints are mounted at runtime (they are not baked into the image):
#   docker build -t tracex .
#   docker run -p 8501:8501 -v $PWD/data:/app/data -v $PWD/checkpoints:/app/checkpoints \
#              -v $PWD/experiments:/app/experiments -v $PWD/reports:/app/reports tracex
#   API instead:  docker run -p 8000:8000 ... tracex uvicorn src.api:app --host 0.0.0.0 --port 8000
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
EXPOSE 8501 8000
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
