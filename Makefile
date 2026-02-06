.PHONY: install-gpu install-mac install-vllm install-dev

install-gpu:
	pip install -e ".[gpu]"
	pip install mlx-lm==0.30.6 "mlx[cpu]==0.30.4" --no-deps

install-mac:
	pip install -e ".[mac]"

install-vllm:
	pip install -e ".[vllm]"

install-dev:
	pip install -e ".[dev]"
