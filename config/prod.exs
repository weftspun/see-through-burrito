import Config

# Burrito executable packaging configuration
# Minimal release: CLI only, no GUI/webserver
config :burrito,
  targets: [
    linux_gnu: [os: :linux, cpu: :x86_64],
    linux_musl: [os: :linux, cpu: :x86_64, libc: :musl],
    macos_x86: [os: :darwin, cpu: :x86_64],
    macos_arm: [os: :darwin, cpu: :aarch64]
  ]
