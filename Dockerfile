FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /benchmark
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[gpu]"

ENTRYPOINT ["global21cm-inference"]
CMD ["0"]
