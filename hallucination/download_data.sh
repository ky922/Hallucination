#!/usr/bin/env bash
# download_data.sh — download POPE annotations and COCO val2014 data
# Run from any directory: bash hallucination/download_data.sh
set -euo pipefail
cd "$(dirname "$0")"

for cmd in wget unzip find wc; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd"
        exit 1
    fi
done

mkdir -p data/pope data/coco/val2014 data/coco/annotations

echo "=== 1/3  Downloading POPE annotation files ==="
# Official POPE repo (only the 3 COCO split JSON files, ~1 MB total)
POPE_BASE="https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
for SPLIT in random popular adversarial; do
    if [ ! -f "data/pope/coco_pope_${SPLIT}.json" ]; then
        echo "  Downloading coco_pope_${SPLIT}.json ..."
        wget -q "${POPE_BASE}/coco_pope_${SPLIT}.json" \
             -O "data/pope/coco_pope_${SPLIT}.json"
    else
        echo "  coco_pope_${SPLIT}.json already exists, skipping."
    fi
done

echo ""
echo "=== 2/3  Downloading COCO val2014 annotations (~241 MB) ==="
if [ ! -f "data/coco/annotations/instances_val2014.json" ]; then
    wget -q --show-progress \
         "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
         -O data/coco/annotations_trainval2014.zip
    echo "  Extracting ..."
    unzip -q data/coco/annotations_trainval2014.zip \
          "annotations/instances_val2014.json" \
          "annotations/captions_val2014.json" \
          -d data/coco/
    rm data/coco/annotations_trainval2014.zip
else
    echo "  instances_val2014.json already exists, skipping."
fi

echo ""
echo "=== 3/3  COCO val2014 images (~13 GB) ==="
# Images are large — download only if the directory is empty.
IMG_COUNT=$(find data/coco/val2014 -name "*.jpg" 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -lt 100 ]; then
    echo "  Downloading COCO val2014 images (this will take a while) ..."
    wget -q --show-progress \
         "http://images.cocodataset.org/zips/val2014.zip" \
         -O data/coco/val2014.zip
    echo "  Extracting ..."
    unzip -q data/coco/val2014.zip -d data/coco/
    rm data/coco/val2014.zip
else
    echo "  Found ${IMG_COUNT} images in data/coco/val2014, skipping download."
fi

echo ""
echo "=== Done ==="
echo "Expected layout:"
echo "  data/pope/coco_pope_{random,popular,adversarial}.json"
echo "  data/coco/annotations/instances_val2014.json"
echo "  data/coco/val2014/COCO_val2014_*.jpg"
