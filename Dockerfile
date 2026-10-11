FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY gaia_agent ./gaia_agent
RUN pip install --no-cache-dir .

# Run as a non-root user; answers and downloaded files go to /data (mount a volume to keep them)
RUN useradd --create-home agent && mkdir -p /data && chown agent /data
USER agent
ENV GAIA_DATA_DIR=/data

ENTRYPOINT ["gaia-agent"]
CMD ["--help"]
