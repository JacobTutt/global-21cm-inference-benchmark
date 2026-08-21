FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /benchmark
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "jax[cuda12]==0.6.2"
COPY . .

ENTRYPOINT ["python", "run_inference.py"]
CMD ["0"]
