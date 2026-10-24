FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Corpus is checked in under rag/processed/corpus.jsonl already, but if it's
# missing (e.g. a fresh clone of just the app/ + scripts/ dirs) build it.
RUN test -f rag/processed/corpus.jsonl || python scripts/build_corpus.py
RUN test -f rag/evaluation/retrieval_gold.jsonl || python scripts/generate_gold_set.py

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
