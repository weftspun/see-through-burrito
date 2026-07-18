#!/bin/bash
# Prepare model weights for release distribution
# Downloads GGUF models from the original sources and compresses with zstd

set -e

RELEASE_DIR="${1:-.}/models_release"
MODEL_BASE="https://huggingface.co"

mkdir -p "$RELEASE_DIR"
cd "$RELEASE_DIR"

echo "Preparing models for release distribution..."
echo ""

# Models to package (model_id, huggingface_path, output_filename)
declare -a MODELS=(
  "layerdiff-unet|shitagaki-lab/layerdiff-unet-gguf|layerdiff-unet.gguf"
  "trans-vae|shitagaki-lab/trans-vae-gguf|trans-vae.gguf"
  "marigold-unet|prs-eth/marigold-gguf|marigold-unet.gguf"
  "sd-vae|stabilityai/sd-vae-ft-mse-gguf|sd-vae.gguf"
  "lama|facebook/lama-gguf|lama.gguf"
  "clip-l|openai/clip-vit-large-patch14-gguf|clip-l.gguf"
  "clip-g|laion/CLIP-ViT-g-14-laion2B-s12B-b42K-gguf|clip-g.gguf"
)

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_id hf_path output_file <<< "$model_spec"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Model: $model_id"
  echo "Path: $hf_path"
  echo ""

  if [ -f "${output_file}.zst" ]; then
    echo "✓ Already compressed: ${output_file}.zst"
    continue
  fi

  if [ ! -f "$output_file" ]; then
    echo "Downloading $model_id..."
    # Use huggingface-hub CLI
    huggingface-cli download "$hf_path" "$output_file" --cache-dir . --local-dir . --local-dir-use-symlinks False
  else
    echo "✓ Already downloaded: $output_file"
  fi

  if [ -f "$output_file" ]; then
    size_mb=$(du -m "$output_file" | cut -f1)
    echo "Compressing with zstd (${size_mb}MB)..."
    zstd -19 -k "$output_file" -o "${output_file}.zst"
    echo "✓ Created ${output_file}.zst"
    echo ""
  fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Ready for release:"
ls -lh *.zst
echo ""
echo "Upload these to GitHub release: https://github.com/weftspun/see-through-burrito/releases/tag/v0.1.0-models"
