FROM python:3.13-slim

ARG MODEL_VERSION=v0.1
ENV MODEL_VERSION=${MODEL_VERSION}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
RUN python -m src.train

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]