.PHONY: install test eval serve clean

install:
	pip install -r requirements.txt

test:
	pytest -q

eval:
	python eval/run_eval.py

serve:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf tickets/*.json __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
