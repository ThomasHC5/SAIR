#!/usr/bin/env bash

set -e
cd "$(dirname "$0")/.."

echo "[1/5] Extracting edges..."
conda run --no-capture-output -n sair python processing_image/preprocessing_edge.py \
    --input_dir Data/Things-EEG2/Image_set/image_set \
    --output_dir Data/Things-EEG2/Image_set/edge_set

echo "[2/5] Estimating depth..."
(
    cd processing_image
    conda run --no-capture-output -n sair-depth python preprocessing_depth.py \
        --input_dir ../Data/Things-EEG2/Image_set/image_set \
        --output_dir ../Data/Things-EEG2/Image_set/depth_set
)

echo "[3/5] Applying foveal blur..."
conda run --no-capture-output -n sair python processing_image/preprocessing_blur.py \
    --input_dir Data/Things-EEG2/Image_set/image_set \
    --output_dir Data/Things-EEG2/Image_set/blur_set

echo "[4/5] Extracting CLIP features..."
for view in edge depth blur; do
    conda run --no-capture-output -n sair python processing_image/preprocessing_clip.py \
        --input_dir "Data/Things-EEG2/Image_set/${view}_set" \
        --output_dir "Data/Things-EEG2/Features_all/${view}"
done

echo "[5/5] Merging per-image features..."
for view in edge depth blur; do
    for split in training test; do
        conda run --no-capture-output -n sair python processing_image/merge_embeddings.py \
            --input_dir "Data/Things-EEG2/Features_all/${view}/${split}_images" \
            --output_path "Data/Things-EEG2/Features/${view}_${split}.npy"
    done
done

echo "Image preprocessing complete."
