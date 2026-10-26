.PHONY: setup download baseline evil census validation 8bit assets report

setup:
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -r requirements.txt

download:
	hf download mlx-community/Qwen3.5-4B-MLX-4bit --local-dir models/Qwen3.5-4B-MLX-4bit
	hf download mlx-community/Qwen3.5-4B-MLX-8bit --local-dir models/Qwen3.5-4B-MLX-8bit

baseline:
	. .venv/bin/activate && python qwen_neuron_steer.py

evil:
	. .venv/bin/activate && python evil_pair_full.py

census:
	. .venv/bin/activate && python activation_census.py

validation:
	. .venv/bin/activate && python persona_validation.py
	. .venv/bin/activate && python negative_control.py

8bit:
	. .venv/bin/activate && python persona_8bit.py

assets:
	. .venv/bin/activate && python make_blog_assets.py
	. .venv/bin/activate && python make_zoo.py

report:
	. .venv/bin/activate && python make_zoo.py
	open zoo.html
