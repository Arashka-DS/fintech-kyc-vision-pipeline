FROM mambaorg/micromamba:1.5.1-bullseye-slim

WORKDIR /app

# Install OS-level dependencies required for OpenCV and EasyOCR
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

COPY . .

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000 8501

# The default command will be overridden by docker-compose
CMD ["micromamba", "run", "-n", "base", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
