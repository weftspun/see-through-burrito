# TODO: API Refinement & Implementation

## API Mismatch Fixes (Blocking)

These functions have API mismatches with actual Bumblebee/Nx versions:
- [ ] `Bumblebee.load_model/2` — check actual signature, may need `:transformers` type or schema
- [ ] `Bumblebee.serve/3` — verify this exists or use alternative (e.g., direct model calls)
- [ ] `Bumblebee.run/3` — verify API or use manual inference
- [ ] `Image.to_binary/2` — use `Image.to_tensor/1` or `Vix.Vips` instead
- [ ] `Nx.expand_dims/2` → use `Nx.new_axis/2`
- [ ] `Nx.max/1`, `Nx.min/1` → use `Nx.reduce_max/1`, `Nx.reduce_min/1` inside defn
- [ ] `Nx.ones/1` → use `Nx.broadcast(1.0, shape)`
- [ ] `Nx.random_normal/2` → use `Nx.Random` module with proper key semantics

## Core Functionality

- [ ] **Model Loading**: Implement safe safetensors loading with Bumblebee
- [ ] **Image Pipeline**: Fix image encoding/tensor conversion
  - [ ] Load images via Vix/Image
  - [ ] Convert to float tensors (0-1 range)
  - [ ] Handle RGBA/transparency correctly
- [ ] **Text Encoding**: Verify CLIP/tokenizer integration
- [ ] **Diffusion Loop**: Implement proper DPM++ or DDIM sampling
  - [ ] Noise scheduling (linear, quadratic, sqrt)
  - [ ] Guidance scaling (classifier-free guidance)
  - [ ] Numerical stability checks
- [ ] **Layer Extraction**: Connect LayerDiff UNet to layer decomposition
- [ ] **Depth Estimation**: Wire Marigold for depth prediction
- [ ] **Inpainting**: Connect LaMa model for hole filling

## Testing

- [ ] Rename `prop_check_tests.exs` → `prop_check_test.exs` (Mix convention)
- [ ] Fix warnings in prop tests (generators, assertions)
- [ ] Add unit tests for each module
- [ ] Add integration tests for full pipeline
- [ ] Benchmark GPU memory usage

## Polish

- [ ] Implement PNG encoding for base64 SVG images (currently placeholder)
- [ ] Add layer export to individual SVG files
- [ ] Generate proper layer manifests with depth metadata
- [ ] Add progress bars for long operations
- [ ] Implement graceful CUDA error handling
- [ ] Add verbose/debug logging modes

## Future Enhancements

- [ ] **Thorvg Rustler NIF**: Advanced vector rendering in SVG (optional)
- [ ] **Parallel Layer Processing**: Multi-threaded/GPU-batched layer extraction
- [ ] **Frame Conditioning**: Leverage LayerDiff's cross-frame dependencies
- [ ] **Burrito Integration**: Re-enable executable packaging when available
- [ ] **Web UI**: warp/Phoenix for interactive layer viewing
- [ ] **PSD Export**: Optional PSD layer format (requires Rustler PSD SDK)

## Known Issues

1. **GPU-only mode**: CUDA not initialized in test environment. Tests may need mocking or skip GPU-dependent code in tests.
2. **Model inference**: Bumblebee APIs need verification against actual 0.7.0 version.
3. **Image handling**: `Image.ex` library has different API than assumed.
4. **defn limitations**: Can't use full Elixir in defn blocks (no recursion, Enum, etc.).
