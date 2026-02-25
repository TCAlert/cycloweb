FROM python:3.11

WORKDIR /app

# System libraries required by cartopy, eccodes, and proj
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Use non-interactive matplotlib backend (no display in container)
ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
