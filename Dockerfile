FROM python:3.11

WORKDIR /app

# System libraries required by cartopy, eccodes, and proj
# fonts-liberation provides Liberation Mono (metrically identical to Courier New)
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libeccodes-dev \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Use non-interactive matplotlib backend (no display in container)
ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Rebuild matplotlib font cache so Liberation Mono is discoverable
RUN python -c "import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
