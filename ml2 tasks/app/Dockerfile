FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

COPY . /app

EXPOSE 8000
#change the host or server to requirements, then build
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]