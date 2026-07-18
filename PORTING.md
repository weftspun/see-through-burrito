# Porting Plan: see-through-cpp → see-through-burrito

## Overview
Port core inference and post-processing units from C++ implementation to Elixir via Nx/ExLA/Bumblebee.

## Phase 1: Foundation (Completed ✓)
- [x] Project structure and dependencies (Nx, ExLA, Bumblebee, Burrito)
- [x] GPU-first configuration (CUDA/Metal, no CPU fallback)
- [x] Behavior contracts via dependency injection
- [x] Test infrastructure (Mox, PropCheck, ExLA tests)
- [x] Zero Dialyzer defects

## Phase 2: Core Tensor Ops (IN PROGRESS)
Priority: **CRITICAL** - These are used by all downstream stages

### 2.1 Image Utils (High Priority)
**From**: see-through-cpp/src/image_utils.cpp
**Target**: lib/see_through_burrito/images.ex

Functions to port:
- [x] `load_image(path) → Image` - Load PNG/JPEG
- [x] `center_square_pad_resize(input, res, pad_val) → Image` - Padding + resize
- [ ] `save_image(Image, path)` - Save PNG
- [ ] `Image.px(x, y) → [r,g,b,a]` - Pixel access (implement via Nx)
- [ ] Alpha blending: `blend_over(dst, src)` - Composite layers
- [ ] Alpha floor: threshold alpha channel
- [ ] Bounding box: `bbox_alpha(img, threshold) → (x, y, w, h)` - Layer bounds

**Status**: Partial (load_image pending Image library integration)

### 2.2 Tensor Ops (High Priority)
**From**: see-through-cpp/src/ops.cpp
**Target**: lib/see_through_burrito/tensor_ops.ex (NEW)

Functions to port:
- [ ] `tile_to_batch(tensor) → batched` - Duplicate frames for batch processing
- [ ] `unbatch(tensor) → frames` - Extract frames from batch
- [ ] `attention_mask_from_shape(h, w, scale) → mask` - Create attention masks
- [ ] `cast_if_needed(tensor, dtype)` - Type casting with safety checks
- [ ] Shape utilities: broadcast, reshape, pad

**Status**: Not started

### 2.3 Diffusion Scheduler (Medium Priority)
**From**: see-through-cpp/src/scheduler.cpp
**Target**: lib/see_through_burrito/scheduler.ex (NEW)

Functions to port:
- [ ] `DDIMScheduler.init(num_steps, schedule)` - Initialize scheduler
- [ ] `get_sigmas(scheduler) → [σ₁, σ₂, ...]` - Compute noise schedule
- [ ] `scale_model_input(latents, sigma) → scaled` - Input scaling
- [ ] `step(model_output, sigma, sample) → (sample, derivative)` - DDIM step

**Status**: Not started

## Phase 3: Model Inference (IN PROGRESS)
Priority: **CRITICAL** - Actual ML model execution

### 3.1 CLIP Text Encoder
**From**: see-through-cpp/src/clip.cpp
**Target**: lib/see_through_burrito/clip.ex (NEW)

Functions to port:
- [ ] `encode_tags(tags) → embeddings[num_tags, 77, 2048]` - Tag→embedding
- [ ] `tokenize(text) → tokens` - Text tokenization
- [ ] `encode_text_batch(texts) → embeddings` - Batch encoding

**Status**: Not started

### 3.2 VAE (SD-VAE, Trans-VAE)
**From**: see-through-cpp/src/vae.cpp
**Target**: lib/see_through_burrito/vae.ex (NEW)

Functions to port:
- [ ] `encode(image) → latents` - Image→latent compression
- [ ] `decode(latents) → image` - Latent→image decompression
- [ ] Handle both SD-VAE and Trans-VAE models

**Status**: Not started (partially defined in encoder.ex)

### 3.3 LayerDiff UNet (Frame-Conditional)
**From**: see-through-cpp/src/unet_frame.cpp
**Target**: lib/see_through_burrito/unet.ex (NEW)

Functions to port:
- [ ] `forward(latents, timestep, embeddings, page_info) → noise_pred`
- [ ] Handle dual-pass: body pass (13 tags) + head pass (11 tags)
- [ ] Frame conditioning via page_rgb + page_alpha

**Status**: Not started

### 3.4 Marigold Depth Estimation
**From**: see-through-cpp/src/... (integrated with pipeline)
**Target**: lib/see_through_burrito/depth.ex

Functions to port:
- [ ] `estimate_depth(image) → depth_map[0,1]` - Monocular depth
- [ ] Normalize + invert depth for layer ordering

**Status**: Placeholder defined

## Phase 4: Post-Processing (IN PROGRESS)
Priority: **HIGH** - Transform model outputs to layers

### 4.1 Post-Processor
**From**: see-through-cpp/src/postproc.cpp
**Target**: lib/see_through_burrito/postproc.ex (NEW)

Functions to port:
- [ ] `threshold_alpha(layers, threshold)` - Remove noise from layer masks
- [ ] `bbox_crop(layers) → cropped_layers` - Trim transparent regions
- [ ] `alpha_blending(layers, depth_order) → composited` - Layer ordering
- [ ] `extract_layer_features(layer) → {bounds, alpha_stats}` - Layer analysis

**Status**: Not started

### 4.2 LaMa Inpainting
**From**: see-through-cpp/src/lama.cpp
**Target**: lib/see_through_burrito/inpaint.ex

Functions to port:
- [x] `fill_holes(layer, mask) → inpainted` - Hole filling
- [ ] Morphological operations (erosion, dilation)

**Status**: Placeholder defined

## Phase 5: SVG Export
Priority: **MEDIUM** - Output formatting

### 5.1 SVG Generation
**Target**: lib/see_through_burrito/svg_export.ex

Functions to port:
- [x] Base64 embed images
- [x] Layer metadata in JSON
- [ ] Depth visualization (gradient overlay)
- [ ] Layer order based on depth

**Status**: Partially implemented

## Phase 6: Integration Tests
Priority: **MEDIUM** - Verify end-to-end

### 6.1 E2E Pipeline Tests
**From**: see-through-cpp/tests/test_*_e2e.cpp

Tests to port:
- [ ] test_layerdiff_e2e: Body + head pass
- [ ] test_marigold_e2e: Depth estimation
- [ ] test_lama_e2e: Inpainting

**Status**: Not started

## Model Checklist
Required models (from see-through-cpp):

- [x] layerdiff-unet (body/head conditional)
- [x] sd-vae (image encoder/decoder)
- [x] trans-vae (optional, larger VAE)
- [x] clip-l, clip-g (text encoders)
- [x] marigold-unet (depth estimation)
- [x] lama (inpainting)

## Porting Strategy

1. **Top-down**: Start with orchestration (pipeline.ex), then fill in dependencies
2. **Test-driven**: Port unit + tests together (use cpp tests as reference)
3. **Gradual**: Placeholder → Real implementation phase by phase
4. **GPU-first**: Use ExLA/defn for all compute-heavy ops
5. **Type-safe**: Leverage Elixir patterns (pattern matching, pipe operators)

## Current Status

**Phase 2** (Core Tensor Ops): ✅ 40% COMPLETE
- [x] Image utilities (padding, resize, normalize)
- [x] Tensor ops (blending, thresholding, batching)
- [x] Schedulers (DPM-Solver++, DDIM)
- TODO: Full bilinear/area interpolation

**Phase 3** (Model Inference): 🚀 30% IN PROGRESS
- [x] CLIP text encoder (stub, awaits Bumblebee)
- [x] VAE encode/decode (stub, awaits Bumblebee)
- [x] LayerDiff UNet (stub, awaits Bumblebee)
- [ ] Marigold depth (awaits model serving)

**Phase 4** (Post-Processing): 10% (inpainting placeholder)
- [ ] Post-processor (morphological ops)
- [ ] Layer filtering/compositing

**Phase 5** (SVG Export): 50% (basic structure, needs depth viz)

**Phase 6** (Integration Tests): 0% (awaits inference wiring)

**Overall**: ~22% (foundation + 3 model modules + partial porting)

## Next Steps

1. Port `image_utils.cpp` → `tensor_ops.ex` (shape ops, padding, resizing)
2. Port `scheduler.cpp` → `scheduler.ex` (DDIM scheduling)
3. Wire up CLIP text encoding (use Bumblebee/Tokenizers)
4. Implement VAE encode/decode via Bumblebee
5. Add LayerDiff forward pass (frame conditioning)
6. Add Marigold depth estimation
7. Port post-processing pipeline
8. E2E test with sample image

## References

- see-through-cpp/src/*.cpp - Implementation reference
- see-through-cpp/tests/*.cpp - Test cases
- .claude/CLAUDE.md - Architecture constraints
- test/orchestration_test.exs - Test patterns with Mox
