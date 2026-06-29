FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 curl git git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .

# Install torch CPU-only first (saves ~1.3GB vs CUDA build)
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    python-multipart==0.0.9 \
    pillow==10.3.0 \
    requests==2.31.0 \
    httpx==0.27.0 \
    "ultralytics==8.2.0" \
    "transformers==4.41.0" \
    "accelerate==0.30.0"

COPY --chown=user:user . .

# Pull actual YOLO model binary if Railway cloned only the Git LFS pointer
RUN if head -c 50 model/yolobv11/best.pt 2>/dev/null | grep -q "version https://git-lfs"; then \
    echo "Pulling YOLO model via git-lfs (Railway skipped LFS during clone)..." && \
    GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 --no-tags \
        https://github.com/trantan66/nutrivision-ai.git /tmp/lfs-repo && \
    cd /tmp/lfs-repo && \
    GIT_TERMINAL_PROMPT=0 git lfs pull --include="model/yolobv11/best.pt" && \
    cp model/yolobv11/best.pt /app/model/yolobv11/best.pt && \
    chown user:user /app/model/yolobv11/best.pt && \
    cd / && rm -rf /tmp/lfs-repo && \
    echo "YOLO model pulled: $(wc -c < /app/model/yolobv11/best.pt) bytes"; \
fi

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
