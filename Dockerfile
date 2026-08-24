FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

# Dependencies first so the layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY data ./data
COPY tests ./tests
COPY Makefile ./

# Build the manifest at image build time so the container starts queryable.
RUN python -m pria.cli ingest

ENTRYPOINT ["python", "-m", "pria.cli"]
CMD ["eval"]
