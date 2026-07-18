import Config

config :logger,
  level: :info

# Burrito configuration for executables
config :burrito,
  targets: [
    linux: [os: :linux, cpu: :x86_64],
    macos: [os: :darwin, cpu: :x86_64]
  ]
