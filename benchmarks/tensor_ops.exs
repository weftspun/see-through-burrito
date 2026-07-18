# Benchmarks for core tensor operations
# Run with: mix run benchmarks/tensor_ops.exs
# Note: GPU-only benchmarks. Requires CUDA or Metal-EXLA to be available.

IO.puts("""
╔════════════════════════════════════════════════════════════════════╗
║ See-Through Burrito - Tensor Operations Benchmarks                ║
║ GPU-Only: Requires CUDA or Metal-EXLA                             ║
╚════════════════════════════════════════════════════════════════════╝
""")

# Check for GPU availability
gpu_available? =
  case EXLA.Defn.default_backend() do
    {EXLA.Backend, [{:client, :cuda} | _]} -> true
    {EXLA.Backend, [{:client, :metal} | _]} -> true
    _ -> false
  end
rescue
  _ -> false

if not gpu_available? do
  IO.puts("⚠️  WARNING: GPU not available. Benchmarks require CUDA or Metal-EXLA.")
  IO.puts("   - On Linux/Windows with NVIDIA GPU: Install CUDA Toolkit")
  IO.puts("   - On macOS with Apple Silicon: Use Metal-EXLA backend")
  IO.puts("")
  IO.puts("Skipping benchmarks.")
  exit(0)
end

IO.puts("✓ GPU available. Setting up benchmark tensors...\n")

small_tensor = Nx.fill(Nx.tensor(0.5), {64, 64, 3})
medium_tensor = Nx.fill(Nx.tensor(0.5), {256, 256, 3})

IO.puts("Starting benchmarks...\n")

Benchee.run(
  %{
    "depth_normalize (64x64)" => fn ->
      SeeThroughBurrito.Depth.normalize_depth(small_tensor)
    end,
    "depth_normalize (256x256)" => fn ->
      SeeThroughBurrito.Depth.normalize_depth(medium_tensor)
    end,
    "depth_invert" => fn ->
      SeeThroughBurrito.Depth.invert_depth(small_tensor)
    end,
    "depth_to_height" => fn ->
      SeeThroughBurrito.Depth.depth_to_height(small_tensor)
    end,
    "image_pad_to_8" => fn ->
      SeeThroughBurrito.Images.pad_to_8_divisible(small_tensor)
    end,
    "image_normalize" => fn ->
      SeeThroughBurrito.Images.normalize(small_tensor)
    end,
    "pipeline_normalize_sdxl" => fn ->
      SeeThroughBurrito.Pipeline.normalize_sdxl(small_tensor)
    end,
    "pipeline_scale_guidance (scale=7.5)" => fn ->
      SeeThroughBurrito.Pipeline.scale_guidance(small_tensor, 7.5)
    end,
    "inpaint_detect_holes" => fn ->
      rgba = Nx.fill(Nx.tensor(0.8), {64, 64, 4})
      SeeThroughBurrito.Inpaint.detect_holes(rgba)
    end,
    "depth_compute_normals" => fn ->
      SeeThroughBurrito.Depth.compute_normals(small_tensor)
    end
  },
  time: 5,
  memory_time: 2,
  warmup: 1,
  formatters: [
    Benchee.Formatters.Console,
    {Benchee.Formatters.HTML, file: "benchmarks/output.html"}
  ],
  print: [
    benchmarking: "Benchmarking :name...",
    configuration: "Configuration:",
    comparison: true,
    fast_warning: true
  ]
)

IO.puts("")
IO.puts("HTML report generated: benchmarks/output.html")
