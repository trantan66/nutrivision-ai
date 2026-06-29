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
    "ultralytics==8.3.0" \
    "transformers==4.41.0" \
    "accelerate==0.30.0"

RUN pip install --no-cache-dir "numpy==1.26.4"

COPY --chown=user:user . .

ARG GITHUB_TOKEN
RUN if head -c 50 model/yolobv11/best.pt 2>/dev/null | grep -q "version https://git-lfs"; then \
    echo "Downloading YOLO model via GitHub LFS API (private repo)..." && \
    LFS_OID=$(grep "^oid sha256:" model/yolobv11/best.pt | sed 's/^oid sha256://') && \
    LFS_SIZE=$(grep "^size " model/yolobv11/best.pt | awk '{print $2}') && \
    echo "OID: $LFS_OID  SIZE: $LFS_SIZE" && \
    DOWNLOAD_URL=$(curl -sf \
        -X POST \
        -H "Content-Type: application/vnd.git-lfs+json" \
        -H "Accept: application/vnd.git-lfs+json" \
        -u "x-access-token:${GITHUB_TOKEN}" \
        -d "{\"operation\":\"download\",\"transfer\":[\"basic\"],\"objects\":[{\"oid\":\"${LFS_OID}\",\"size\":${LFS_SIZE}}]}" \
        "https://github.com/trantan66/nutrivision-ai.git/info/lfs/objects/batch" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d['objects'][0]['actions']['download']['href'])") && \
    curl -fL -o model/yolobv11/best.pt "$DOWNLOAD_URL" && \
    chown user:user model/yolobv11/best.pt && \
    echo "YOLO model downloaded: $(wc -c < model/yolobv11/best.pt) bytes"; \
fi

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
