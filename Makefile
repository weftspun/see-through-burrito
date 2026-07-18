.PHONY: help test bench build release clean

help:
	@echo "see-through-burrito - Elixir ML anime layer decomposition"
	@echo ""
	@echo "Commands:"
	@echo "  make test      - Run unit tests"
	@echo "  make test-gpu  - Run GPU tests (requires CUDA/Metal)"
	@echo "  make bench     - Run performance benchmarks"
	@echo "  make build     - Compile the project"
	@echo "  make escript   - Build command-line executable"
	@echo "  make release   - Build release executable"
	@echo "  make clean     - Clean build artifacts"

test:
	mix test

test-gpu:
	mix test --include skip

bench: build
	@echo ""
	@echo "Running benchmarks..."
	@echo "Note: GPU benchmarks require CUDA/Metal to be available"
	mix run benchmarks/tensor_ops.exs

build:
	mix compile

escript: build
	mix escript.build
	@echo "✓ Built executable: ./see_through_burrito"

release: build
	@echo "Building release executable with Burrito..."
	@echo "Note: Requires Burrito package (optional dependency)"
	@echo "Skipped: Burrito not available on Hex.pm"

clean:
	mix clean
	rm -f ./see_through_burrito
	rm -rf benchmarks/output.html
	find . -name "*.swp" -delete
