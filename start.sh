#!/bin/sh
set -e

MODEL_FILE="model/yolobv11/best.pt"

# Detect Git LFS pointer (Railway does not pull LFS binaries by default)
if [ -f "$MODEL_FILE" ] && head -c 50 "$MODEL_FILE" | grep -q "version https://git-lfs"; then
    echo "Git LFS pointer detected for $MODEL_FILE — downloading actual model binary..."

    LFS_OID=$(grep "oid sha256:" "$MODEL_FILE" | sed 's/oid sha256://')
    LFS_SIZE=$(grep "^size " "$MODEL_FILE" | awk '{print $2}')

    echo "OID: $LFS_OID  SIZE: $LFS_SIZE"

    BATCH_RESPONSE=$(curl -sf \
        -X POST \
        -H "Content-Type: application/vnd.git-lfs+json" \
        -H "Accept: application/vnd.git-lfs+json" \
        -d "{\"operation\":\"download\",\"transfer\":[\"basic\"],\"objects\":[{\"oid\":\"${LFS_OID}\",\"size\":${LFS_SIZE}}]}" \
        "https://github.com/trantan66/nutrivision-ai.git/info/lfs/objects/batch")

    DOWNLOAD_URL=$(printf '%s' "$BATCH_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['objects'][0]['actions']['download']['href'])
")

    echo "Downloading model from GitHub LFS CDN..."
    curl -fL -o "$MODEL_FILE" "$DOWNLOAD_URL"
    echo "Model downloaded successfully ($(wc -c < $MODEL_FILE) bytes)"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
