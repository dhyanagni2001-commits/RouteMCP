FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROUTEMCP_CONFIG=/app/models.yaml \
    ROUTEMCP_DB=/data/routemcp.db

RUN addgroup --system routemcp && adduser --system --ingroup routemcp routemcp
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[mcp]'
COPY models.example.yaml ./models.yaml
RUN mkdir /data && chown routemcp:routemcp /data

USER routemcp
EXPOSE 8000 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["routemcp", "serve", "--config", "/app/models.yaml", "--host", "0.0.0.0", "--port", "8000"]
